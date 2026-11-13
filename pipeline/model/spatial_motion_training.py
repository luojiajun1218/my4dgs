"""逐帧空间共享运动的训练与协调。

组织区域运动、逐点残差和 gate 的优化时序，并在区域切换时衔接运动表示；
gate 及候选区域放在 spatial_region_gate.py，属性更新和存储估算交给对应模块。
"""

from pathlib import Path
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

from .rigid_residual_tree import (MotionBase, _id_lengths, _point_id_lengths, _integer,
    _quantized, quaternion_multiply,
    rotation_vector_quaternion)
from .spatial_region_gate import (SpatialGateController, spatial_layout, spatial_owner, spatial_state_route,
                          SPATIAL_ROUTING_FIELDS)
from .residual_gs_attributes_update import ResidualGSMotionAttributes
from .regional_attributes_update import (RegionalMotionAttributes, _HardForwardGateProxy,
    _hard_value_soft_gradient, _rate_relative_representative, _retain_bin_remainder)


def predictor_cost(target, predictor, metric, steps, l1, byte_price, row_bytes):
    """Return per-point envelope cost and the chosen quantized residual.

    Try the same T/R block support choices as GS proximal sparsification. Hold
    the chosen residual fixed in backward (the envelope derivative); only the
    predictor learns here. Gates receive these costs through path probabilities.
    """
    target, metric = target.detach(), metric.detach()
    with torch.no_grad():
        difference = target - predictor.detach()
        residual = difference.sign() * (difference.abs() - l1 / metric.clamp_min(1e-20)).clamp_min(0)
        residual = (residual / steps).round().clamp(-32767, 32767) * steps
        options = torch.stack((residual, torch.cat((residual[:, :3], residual[:, 3:] * 0), -1),
            torch.cat((residual[:, :3] * 0, residual[:, 3:]), -1), residual * 0))
        active_t = options[:, :, :3].ne(0).any(-1)
        active_r = options[:, :, 3:].ne(0).any(-1)
        from .storage_rate import raw_motion_block_bytes
        byte_cost = raw_motion_block_bytes(active_t.int(), active_r.int(), row_bytes)
        local = (.5 * metric * (predictor.detach()[None] + options - target[None]).square()).sum(-1)
        local += (l1 * options.abs()).sum(-1) + byte_price * byte_cost
        chosen = local.argmin(0)
        rows = torch.arange(len(target), device=target.device)
        selected = options[chosen, rows]
        selected_bytes = byte_cost[chosen, rows]
    distortion = (.5 * metric * (predictor + selected - target).square()).sum(-1)
    penalty = (l1 * selected.abs()).sum(-1)
    return distortion + penalty + byte_price * selected_bytes, selected, selected_bytes










class SpatialMotionTraining(SpatialGateController, MotionBase):
    """Learn region increments and GS residuals; gates select spatial resolution."""
    @property
    def joint_training(self):
        return self.config.spatial_gate.training_schedule == 'joint'

    @property
    def proximal_gs_regularization(self):
        return self.joint_training and self.config.proximal_gs_regularization

    @property
    def joint_image_gradients(self):
        return self.joint_training and self.config.spatial_gate.joint_image_gradients

    @property
    def relative_regions(self):
        return self.config.spatial_gate.region_motion == 'relative'

    @property
    def hierarchical_private(self):
        return getattr(self, '_frame_hierarchical_private', self.config.spatial_gate.hierarchical_private)

    @property
    def private_full_motion(self):
        return getattr(self, '_frame_private_full_motion', self.config.spatial_gate.private_full_motion)

    @property
    def all_region_proposals(self):
        return self.private_full_motion and not self.hierarchical_private

    @property
    def shared_discovery_active(self):
        return self.config.spatial_gate.shared_discovery and (
            self.last_step <= self.config.shared_discovery_steps if self.config.spatial_gate.frame_training_plan else not self._hard_gate_frozen)

    @property
    def refining_unselected_candidates(self):
        cfg=self.config.spatial_gate
        return (cfg.frame_training_plan and cfg.refine_unselected_candidates and
                cfg.candidate_refinement_start_step <= self.last_step <= 180)

    @torch.no_grad()
    def prepare_plan_step(self, step):
        if not self.config.spatial_gate.frame_training_plan:
            return
        previous_step=self.last_step
        self.last_step = int(step)
        self._prepare_region_gate_step(step, previous_step)
        tail_steps = self.config.spatial_gate.final_joint_refine_steps
        motion = step <= 200 or bool(tail_steps and 301-tail_steps <= step <= 300)
        for parameter, enabled in ((self.region_translation, motion),
                (self.region_rotation, motion and self.config.rotation_updates),
                (self.gs_translation, motion and step > self.config.shared_discovery_steps),
                (self.gs_rotation, motion and step > self.config.shared_discovery_steps and self.config.rotation_updates)):
            parameter.requires_grad_(enabled)
            if not enabled:
                parameter.grad = None
        self._enforce_rotation_policy()

    def _motion_fit_window_due(self):
        if (self.config.spatial_gate.frame_training_plan and
                self.config.spatial_gate.final_joint_refine_steps and self.last_step == 300):
            return True
        return super()._motion_fit_window_due()

    def _private_nodes(self):
        return self.spatial_paths[:, -1] if self.hierarchical_private else self.owner

    def _private_origins(self):
        leaf = self._private_nodes()
        return tuple(value[leaf].detach() for value in self._regional_predictions())

    def _private_residuals(self, detach_proposal=False):
        return self.residual_motion._private_residuals(self, detach_proposal=detach_proposal)

    def _private_symbols(self):
        return self.residual_motion._private_symbols(self)







    def _stochastic_image_motion(self):
        weights = self._sampled_path_weights()
        result = []
        for region, proposal, grid in zip(self._regional_predictions(),
                (self.gs_translation, self.gs_rotation), (self.gs_steps[:3], self.gs_steps[3:])):
            private = _quantized(proposal, grid)
            experts = (torch.cat((region[self.spatial_paths[:, :-1]], private[:, None]), 1)
                       if self.private_full_motion else region[self.spatial_paths])
            # Parameter gradients use the actually sampled expert; only the
            # discrete selection uses the relaxed path derivative.
            field = (weights[..., None] * experts).sum(1)
            if not self.private_full_motion:
                field = field + (weights[:, -1:] * private if self.hierarchical_private else private)
            result.append(field)
        return tuple(result)


    def _private_support(self):
        """All GS may correct motion unless hierarchical permission is enabled."""
        if not self.hierarchical_private:
            return torch.ones_like(self.owner, dtype=torch.bool)
        return self.owner.eq(self.spatial_paths[:, -1])

    def _private_probabilities(self):
        if not self.hierarchical_private:
            return torch.ones_like(self.gs_translation[:, 0])
        if self._hard_gate_frozen:
            return self._private_support().to(self.gs_translation)
        return self._gate_probabilities()[self.spatial_paths[:, :-1]].prod(-1)

    def _gs_motion(self, hard=False, relaxed=False):
        return self.residual_motion._gs_motion(self, hard=hard, relaxed=relaxed)

    @torch.no_grad()
    def _finalize_private_support(self):
        """Freeze physical capacity and start moments for its new image Jacobian."""
        support = self._private_support()
        for parameter in (self.gs_translation, self.gs_rotation):
            parameter[~support] = 0
        # Before freezing, GS/candidate image gradients had soft path weights.
        # Afterwards selected parameters receive the full physical derivative.
        for parameter in (self.region_translation, self.region_rotation,
                          self.gs_translation, self.gs_rotation):
            self.optimizer.state.pop(parameter, None)

    @torch.no_grad()
    def _begin_fixed_gate_residuals(self):
        """Retain encoded motion while making shared leaf fields physical again."""
        if not self._hard_gate_frozen or not self.private_full_motion:
            raise ValueError('Residual coordinates require newly frozen full-motion proposals')
        support = self._private_support()
        symbols = self._private_symbols()
        for parameter, integer, grid in zip((self.gs_translation, self.gs_rotation),
                symbols, (self.gs_steps[:3], self.gs_steps[3:])):
            parameter.copy_(_retain_bin_remainder(integer.to(parameter), parameter, grid))
            parameter[~support] = 0
        self._frame_private_full_motion = False
        self.flat_region_diagnostics['post_gate_parameterization'] = dict(
            step=self.last_step, mode='shared_plus_private_residual', preserves_encoded_motion=True)

    @property
    def shared_image_recode(self):
        return self.joint_training and self.config.spatial_gate.shared_image_recode

    @property
    def gs_sparsity_active(self):
        return (not self.config.defer_gs_sparsity_until_gate_freeze or
                self.last_step > self.config.spatial_gate.freeze_step)

    @property
    def amplitude_refit_active(self):
        return getattr(self, '_amplitude_refit_support', None) is not None

    @property
    def regional_image_updates(self):
        return (self.amplitude_refit_active or self.shared_image_recode or
                (self.config.spatial_gate.predictor_objective == 'proximal_rd' and self._hard_gate_frozen))

    @torch.no_grad()
    def _prepare_amplitude_refit(self, step):
        count = self.config.amplitude_refit_steps
        if not count or self.amplitude_refit_active or step <= self.motion_steps - count:
            return
        if not self._hard_gate_frozen:
            raise ValueError('Freeze learned gates before motion amplitude refitting')
        self._amplitude_refit_support = {}
        stored = self._stored_region_support()
        for name, grid in (('region_translation', self.steps[:3]),
                ('region_rotation', self.steps[3:]), ('gs_translation', self.gs_steps[:3]),
                ('gs_rotation', self.gs_steps[3:])):
            parameter = getattr(self, name)
            support = torch.round(parameter / grid).ne(0).any(-1)
            if name.startswith('region_'):
                support &= stored
                # Unselected dictionary entries remain available next frame.
                parameter[stored & ~support] = 0
                self.optimizer.state.pop(parameter, None)
            else:
                parameter[~support] = 0
                for key in ('exp_avg', 'exp_avg_sq'):
                    state = self.optimizer.state.get(parameter, {})
                    if key in state:
                        state[key].mul_(support[:, None])
            self._amplitude_refit_support[name] = support

    @torch.no_grad()
    def _mask_amplitude_refit_updates(self):
        for name, support in self._amplitude_refit_support.items():
            parameter = getattr(self, name)
            if parameter.grad is not None:
                parameter.grad.mul_(support[:, None])
            for key in ('exp_avg', 'exp_avg_sq'):
                state = self.optimizer.state.get(parameter, {})
                if key in state:
                    state[key].mul_(support[:, None])

    def _regional_predictions(self, hard=False):
        return self.regional_motion.predictions(self, hard=hard)

    def _stored_region_support(self):
        return self.regional_motion.stored_support(self)

    @torch.no_grad()
    def enable_relative_regions(self, *, inherit_unused=False):
        return self.regional_motion.enable_relative(self, inherit_unused=inherit_unused)

    def initialize_reference(self, xyz, rotation, extent, frame_index=0):
        from .frame_stream_format import (DecodedState, _validate_state)
        cfg=self.config.spatial_gate
        if cfg.reference_layout is None:
            paths, parents, opened, centers, routing = spatial_layout(
                xyz.detach(), cfg.max_depth, return_centers=True,
                initialization=cfg.initialization,
                distance_quantile=cfg.initialization_distance_quantile,
                outlier_assignment=cfg.outlier_assignment,
                return_routing=True)
        else:
            # Compaction changes the GS population, not the predefined
            # physical candidate boundaries. The new reference embeds all
            # routing arrays; decoding never needs the source template.
            with np.load(cfg.reference_layout,allow_pickle=False) as packet:
                template=DecodedState.from_reference(packet)
                source_extent=float(packet['extent'])
            _validate_state(template)
            if (frame_index!=0 or template.frame_index!=0 or
                    template.spatial_paths is None or template.spatial_centers is None or
                    template.spatial_paths.shape[1]!=cfg.max_depth):
                raise ValueError('reference_layout must be a frame0 reference at the same spatial depth')
            if not np.isclose(source_extent,float(extent),rtol=1e-6,atol=1e-8):
                raise ValueError('reference_layout extent differs from the current coordinate system')
            nearest=template.spatial_support_ball is not None
            if nearest!=(cfg.outlier_assignment=='nearest_region'):
                raise ValueError('reference_layout outlier routing differs from configuration')
            paths=spatial_state_route(xyz.detach(),template,cfg.max_depth)
            parents=template.parents.clone()
            centers=template.spatial_centers.clone()
            opened=torch.zeros_like(template.spatial_open)
            opened[0]=True
            routing={name:(None if getattr(template,name) is None else getattr(template,name).clone())
                     for name in SPATIAL_ROUTING_FIELDS}
        owner = spatial_owner(paths, opened)
        self.codec_previous_state = DecodedState(
            xyz.detach().cpu().float().clone(), rotation.detach().cpu().float().clone(),
            owner.cpu(), parents.cpu(), frame_index,
            spatial_paths=paths.cpu(), spatial_open=opened.cpu(), spatial_centers=centers,
            **routing)
        _validate_state(self.codec_previous_state)
        return self.codec_previous_state

    def setup(self, xyz, rotation, extent, frame_index=0, previous_rows=None):
        if self.codec_previous_state is None:
            self.initialize_reference(xyz, rotation, extent, frame_index - 1)
        prior = self.codec_previous_state
        if prior.spatial_paths is None or frame_index != prior.frame_index + 1:
            raise ValueError('Spatial motion requires the immediately preceding spatial state')
        rows = (torch.arange(len(xyz)) if previous_rows is None else
                torch.as_tensor(previous_rows).cpu().long())
        if (rows.shape != (len(xyz),) or len(rows.unique()) != len(rows) or
                bool(((rows < 0) | (rows >= len(prior.xyz))).any())):
            raise ValueError('Invalid spatial survivor rows')
        if not torch.allclose(xyz.detach().cpu(), prior.xyz[rows], atol=1e-6, rtol=1e-5):
            raise ValueError('Spatial motion reference differs from the previous decoded frame')
        # Reuse the last applied increment only as the next free-fit starting
        # point. Sharing, gates and residual sparsification are still fitted
        # to the current images. No additional motion field is transmitted.
        proposal = None
        previous_regions = previous_gates = previous_residuals = None
        if self.committed and self.frame_index == frame_index - 1:
            with torch.no_grad():
                t, r = self._full_motion_symbols()
                sources = self.pending_birth_sources.to(t.device)
                t = torch.cat((t, t[sources]), dim=0)
                r = torch.cat((r, r[sources]), dim=0)
                if len(t) != len(prior.xyz):
                    raise ValueError('Previous motion proposal lost GS lineage')
                indices = rows.to(t.device)
                # Native ReCon-GS retains .6 of translation; its normalized
                # quaternion update retains the previous orientation increment.
                if self.joint_training:
                    proposal = (t[indices], r[indices])
                    previous_regions = (self.region_translation.detach().clone(),
                                        self.region_rotation.detach().clone())
                    previous_gates = self.gate_logits.detach().clone()
                    previous_residuals = tuple(torch.cat((value.detach(), value.detach()[sources]))[indices]
                        for value in (self.gs_translation, self.gs_rotation))
                else:
                    proposal = (torch.round(.6 * t[indices]).clamp(-32767, 32767),
                                r[indices].clamp(-32767, 32767))
        # A frame-boundary migration must read the preceding frame's applied
        # motion under its original policy before enabling the new capacity.
        self._frame_hierarchical_private = self.config.spatial_gate.hierarchical_private
        self._frame_private_full_motion = self.config.spatial_gate.private_full_motion
        self.base_xyz, self.base_rotation = prior.xyz[rows].to(xyz), prior.rotation[rows].to(rotation)
        self.extent, self.frame_index = float(extent), int(frame_index)
        self.previous_rows = rows.to(xyz.device)
        self.frame_reference_state = prior
        self._survivor_mapping_bytes = self._count_survivor_mapping_bytes()
        if prior.spatial_centers is None:
            raise ValueError('Fixed-membership spatial references are replay-only')
        self.spatial_paths = spatial_state_route(prior.xyz[rows], prior,
            self.config.spatial_gate.max_depth).to(xyz.device)
        self.parents = prior.parents.to(xyz.device)
        self.opened = prior.spatial_open.to(xyz.device).clone()
        self.owner = spatial_owner(self.spatial_paths, self.opened)
        count = len(self.parents)
        self._initialize_region_gates(xyz, count, previous_gates, frame_index)
        self.regional_motion = RegionalMotionAttributes(xyz, count)
        self.residual_motion = ResidualGSMotionAttributes(xyz)
        if proposal is not None:
            with torch.no_grad():
                self.gs_translation.copy_(proposal[0].to(xyz) * self.gs_steps[:3])
                self.gs_rotation.copy_(proposal[1].to(xyz) * self.gs_steps[3:])
        self._make_optimizer()
        self.rate_reference_count = self.rate_reference_count or len(xyz)
        self._region_id_lengths = _id_lengths(count - 1, xyz)
        self._gs_id_lengths = _point_id_lengths(len(xyz), xyz)
        self.observations = torch.zeros(len(xyz), dtype=torch.long, device=xyz.device)
        self.pending_birth_sources = None
        self.topology_edits, self.last_metrics = [], {}
        self._codec_rate = None
        self.committed = False
        self.last_step = 0
        if self.stochastic_image_gates:
            self._initialize_gate_sampler()
        self._candidate_initialized = False
        self._shared_field_active = False
        self._regional_fit_support = None
        self._coding_initialized = False
        self._hard_gate_frozen = False
        self._amplitude_refit_support = None
        self._predictor_fit_state = None
        self.flat_region_diagnostics = {'method': 'learned_spatial_gate', 'frame': frame_index}
        self.owner = spatial_owner(self.spatial_paths, self.opened)
        if self.joint_training:
            self._begin_joint_frame(proposal, previous_regions, previous_residuals)
        self._enforce_rotation_policy()

    @torch.no_grad()
    def _enforce_rotation_policy(self):
        """Freeze frame-to-frame orientation increments, keeping the base pose.

        Applied after warm starts so an inherited increment cannot leak into
        a translation-only frame. Newborn attributes retain their normal fit.
        """
        if self.config.rotation_updates:
            return
        for parameter in (self.region_rotation, self.gs_rotation):
            parameter.zero_()
            parameter.grad = None
            parameter.requires_grad_(False)
            self.optimizer.state.pop(parameter, None)

    @torch.no_grad()
    def _begin_joint_frame(self, proposal, previous_regions, previous_residuals):
        """Fit from step one, retaining the previous frame's actual increments."""
        if self.frame_index == 1:
            # The initial logit's sign selects the first hard frontier. A
            # negative value starts with the eight coarse regions; later
            # frames keep their inherited decisions and logits.
            self.opened = self._gate_probabilities().ge(.5)
            self.opened[0] = True
        self.owner = spatial_owner(self.spatial_paths, self.opened)
        if previous_regions is not None:
            for parameter, previous in zip((self.region_translation, self.region_rotation), previous_regions):
                if parameter.shape != previous.shape:
                    raise ValueError('Warm-started joint motion requires the same candidate tree')
                parameter.copy_(previous.to(parameter))
        if proposal is not None:
            rt, rr = (value.to(self.gs_translation) for value in self._regional_predictions(hard=True))
            residuals = (proposal if self.private_full_motion else
                         (proposal[0].to(rt) - rt[self.owner], proposal[1].to(rr) - rr[self.owner]))
            for parameter, symbols, remainder, step in zip(
                    (self.gs_translation, self.gs_rotation), residuals,
                    previous_residuals, (self.gs_steps[:3], self.gs_steps[3:])):
                if bool(symbols.abs().gt(32767).any()):
                    raise ValueError('Joint warm-start exceeds the int16 residual range')
                parameter.copy_(_retain_bin_remainder(symbols, remainder.to(parameter), step))
        self._candidate_initialized = self._coding_initialized = True
        self._shared_field_active = True
        self._regional_fit_support = self.node_counts.gt(0)
        self._regional_fit_support[0] = False
        self.flat_region_diagnostics.update(training_schedule='joint',
            temporal_initialization='previous_full_increment' if proposal is not None else 'zero',
            initial_gates=(('all_eligible_open' if self.opened[1:].any() else 'all_eligible_closed')
                           if self.frame_index == 1 else 'previous_frame'))
        if self.config.spatial_gate.frame_training_plan:
            # Only previously transmitted relative corrections are inherited.
            # Newly available children start at their parent, once per frame.
            prior = self.frame_reference_state
            used = torch.bincount(prior.owner.to(self.owner.device), minlength=len(self.parents)).gt(0)
            nodes = used.nonzero(as_tuple=True)[0]
            for _ in range(self.spatial_paths.shape[1]):
                used[nodes] = True
                nodes = self.parents[nodes].clamp_min(0)
            self.region_translation[~used] = 0
            self.region_rotation[~used] = 0
        if self.config.spatial_gate.shared_discovery:
            # Start from inherited regional predictions. Point moments learn
            # image sensitivity while zero learning rates hold E at zero.
            self.gs_translation.zero_()
            self.gs_rotation.zero_()
        if self.config.spatial_gate.frame_training_plan:
            self.prepare_plan_step(1)

    def _make_optimizer(self):
        super()._make_optimizer()
        self.optimizer.add_param_group(self._region_gate_optimizer_group())

    @property
    def shared_prestructure_active(self):
        return True


    @torch.no_grad()
    def _mask_regional_fit_updates(self):
        return self.regional_motion.mask_fit_updates(self)








    def _values(self, hard=False, step=None):
        rt, rr = self.regional_motion.values(self, hard=hard)
        gt, gr = self._gs_motion(hard=hard)
        return rt, rr, gt, gr

    def _shared_motion(self, relaxed=False):
        return self.regional_motion.shared_motion(self, relaxed=relaxed)

    @property
    def proposal_image_weight(self):
        return (self.config.spatial_gate.proposal_image_weight
                if self.private_full_motion and not self._hard_gate_frozen else 0.)

    def proposal_pose(self):
        """Fit complete proposals even when the sampled gate chooses sharing.

        This training-only field has no regional or gate dependence. The
        final pose and wire packet always use the learned discrete frontier.
        """
        if not self.private_full_motion or self._hard_gate_frozen:
            raise ValueError('Proposal images require unfrozen complete-motion coordinates')
        xyz = self.base_xyz + _quantized(self.gs_translation, self.gs_steps[:3])
        if not self.config.rotation_updates:
            return xyz, self.base_rotation
        vector = _quantized(self.gs_rotation, self.gs_steps[3:])
        return xyz, quaternion_multiply(rotation_vector_quaternion(vector), self.base_rotation)

    @property
    def predictor_image_weight(self):
        return (self.config.spatial_gate.predictor_image_weight
                if (self.all_region_proposals or self.shared_image_recode) and not self._hard_gate_frozen else 0.)

    def predictor_pose(self):
        """Render one actual sampled shared predictor, without GS corrections."""
        if not self.predictor_image_weight:
            raise ValueError('Predictor images require unfrozen all-region proposals')
        weights = self._sampled_path_weights(sample_if_needed=False)
        t, r = ((weights[..., None] * value[self.spatial_paths]).sum(1)
                for value in self._regional_predictions())
        rotation = (quaternion_multiply(rotation_vector_quaternion(r), self.base_rotation)
                    if self.config.rotation_updates else self.base_rotation)
        return self.base_xyz + t, rotation

    def forward(self, step, training=False):
        if training:
            # The frame-plan controller normally prepares the step before
            # rendering. Preserve standalone/non-plan calls as well.
            if not self.config.spatial_gate.frame_training_plan:
                self.last_step = int(step)
                self.prepare_plan_step(step)
            elif self.last_step != int(step):
                self.prepare_plan_step(step)
            self._prepare_amplitude_refit(step)
        if training and self.shared_discovery_active:
            translation, rotation_vector = self._stochastic_image_motion()
            xyz = self.base_xyz + translation
        elif self.all_region_proposals or self.predictor_image_weight and self.shared_image_recode:
            if training:
                self._sampled_path_weights()
            shared_t, shared_r = self._shared_motion()
            gt, gr = self._gs_motion()
            xyz = self.base_xyz + shared_t + gt
            rotation_vector = shared_r + gr
        elif (training and self.stochastic_image_gates and not self._hard_gate_frozen
              and not self.config.spatial_gate.frame_training_plan):
            translation, rotation_vector = self._stochastic_image_motion()
            xyz = self.base_xyz + translation
        else:
            relaxed = training and self.soft_image_active
            shared_t, shared_r = self._shared_motion(relaxed=relaxed)
            gt, gr = self._gs_motion(relaxed=relaxed)
            xyz = self.base_xyz + shared_t + gt
            rotation_vector = shared_r + gr
            if (training and getattr(self, 'refine_image_gate_ste', False) and
                    self.config.spatial_gate.frame_training_plan and
                    self.config.spatial_gate.joint_gate_start_step <= step <= 180):
                if self._hard_gate_frozen or not self.shared_image_recode or self.hierarchical_private or self.private_full_motion:
                    raise ValueError('Gate image STE requires the unfrozen physical-recoding frame plan')
                # A path's alternatives use each candidate's quantized region
                # motion plus the same quantized point residual. Only the
                # gate receives this surrogate derivative; all motion-parameter
                # derivatives remain those of the original hard forward.
                weights = self._path_weights(self._gate_probabilities())
                rt, rr = self._regional_predictions()
                soft_t = (weights[..., None] * (rt[self.spatial_paths].detach() + gt.detach()[:, None, :])).sum(1)
                xyz = _HardForwardGateProxy.apply(xyz, soft_t)
                if self.config.rotation_updates:
                    soft_r = (weights[..., None] * (rr[self.spatial_paths].detach() + gr.detach()[:, None, :])).sum(1)
                    rotation_vector = _HardForwardGateProxy.apply(rotation_vector, soft_r)
        if not self.config.rotation_updates:
            return xyz, self.base_rotation
        rotation = quaternion_multiply(rotation_vector_quaternion(rotation_vector), self.base_rotation)
        return xyz, rotation

    def _full_motion_symbols(self):
        shared_t, shared_r = self._shared_motion()
        if self.private_full_motion:
            gt, gr = self._private_symbols()
            return (torch.round(shared_t / self.steps[:3]) + gt.to(torch.int32),
                    torch.round(shared_r / self.steps[3:]) + gr.to(torch.int32))
        gt = torch.round(self.gs_translation / self.gs_steps[:3])
        gr = torch.round(self.gs_rotation / self.gs_steps[3:])
        if self.hierarchical_private:
            mask = self._private_support()[:, None]
            gt, gr = gt * mask, gr * mask
        return (torch.round(shared_t / self.steps[:3]) + gt,
                torch.round(shared_r / self.steps[3:]) + gr)

    @torch.no_grad()
    def _rebase_full_motion(self, full_motion, preserve_remainder=False):
        shared_t, shared_r = self._shared_motion()
        t = full_motion[0] - torch.round(shared_t / self.steps[:3])
        r = full_motion[1] - torch.round(shared_r / self.steps[3:])
        if t.abs().max() > 32767 or r.abs().max() > 32767:
            raise ValueError('Spatial recoding exceeds int16 residual range')
        if preserve_remainder:
            self.gs_translation.copy_(_retain_bin_remainder(t, self.gs_translation, self.gs_steps[:3]))
            self.gs_rotation.copy_(_retain_bin_remainder(r, self.gs_rotation, self.gs_steps[3:]))
        else:
            self.gs_translation.copy_(t * self.gs_steps[:3])
            self.gs_rotation.copy_(r * self.gs_steps[3:])

    @torch.no_grad()
    def retain_survivors(self, rows):
        """Retire fitted low-alpha GS before births, preserving previous IDs."""
        if self.committed:
            raise ValueError('Cannot retire GS after motion packet finalization')
        rows = torch.as_tensor(rows, device=self.base_xyz.device, dtype=torch.long)
        count = len(self.owner)
        if (rows.ndim != 1 or not len(rows) or
                bool(((rows < 0) | (rows >= count)).any()) or
                bool((rows[1:] <= rows[:-1]).any())):
            raise ValueError('Motion survivors must be nonempty ordered row IDs')
        for name in ('owner', 'base_xyz', 'base_rotation', 'previous_rows',
                     'observations', 'spatial_paths'):
            setattr(self, name, getattr(self, name)[rows].clone())
        self.residual_motion.retain(rows, self.optimizer)
        if self.amplitude_refit_active:
            for name in ('gs_translation', 'gs_rotation'):
                self._amplitude_refit_support[name] = self._amplitude_refit_support[name][rows].clone()
        if getattr(self, '_predictor_fit_state', None) is not None:
            self._predictor_fit_state = {name: value[rows].clone()
                                        for name, value in self._predictor_fit_state.items()}
        self.node_counts = torch.bincount(self.spatial_paths.flatten(), minlength=len(self.parents))
        self._gs_id_lengths = _point_id_lengths(len(rows), self.base_xyz)
        self._survivor_mapping_bytes = self._count_survivor_mapping_bytes()

    def _count_survivor_mapping_bytes(self):
        from .storage_rate import survivor_mapping_bytes
        return survivor_mapping_bytes(self)

    def _metadata_bytes(self):
        from .storage_rate import motion_metadata_bytes
        return motion_metadata_bytes(self)

    def regularization_loss(self, step):
        """Camera-independent training penalty; free proposals have zero price."""
        if not self._candidate_initialized or self.amplitude_refit_active:
            return self.base_xyz.new_zeros(())
        terms = self.loss_terms(step, diagnostics=False)
        return terms['rate_loss'] + terms['relative_motion_loss'] + terms.get('candidate_motion_loss', 0.)

    @torch.no_grad()
    def _measure_codec_rate(self, step, force=False):
        """Encode the current old-point update, without unknown future births."""
        from .storage_rate import MotionCodecRate
        from .frame_stream_encode import (encode_motion)
        rate = getattr(self, '_codec_rate', None)
        rebuild = rate is None or rate.points != len(self.owner)
        due = (force or rebuild or step <= self.config.codec_rate_initial_steps
               or step in (1, self.config.spatial_gate.freeze_step, self.motion_steps)
               or step % self.config.codec_rate_interval == 0)
        if not due and rate.tables is not None:
            return None, rate.last_scale
        rt, rr, gt, gr = [value.cpu() for value in self._values(hard=True)]
        if rebuild:
            empty = encode_motion(self.frame_reference_state, self.frame_index, self.extent,
                self.previous_rows.cpu(), self.owner.cpu(), self.parents.cpu(),
                torch.zeros_like(rt), torch.zeros_like(rr), torch.zeros_like(gt), torch.zeros_like(gr),
                torch.empty(0, dtype=torch.long), torch.empty(0, 3), torch.empty(0, 4),
                self.config.motion_xyz_step_extent, self.config.motion_rotation_step_rad,
                compression='none', gs_translation_step_extent=self.config.gs_xyz_step_extent,
                gs_rotation_step_rad=self.config.gs_rotation_step_rad, spatial_open=self.opened.cpu(),
                spatial_relative=self.relative_regions)
            rate = self._codec_rate = MotionCodecRate(empty, len(self.parents), len(self.owner),
                self.config.motion_fit_window_steps or 20)
        actual = rate.measure(rt, rr, gt, gr, self.opened.cpu(), int(step))
        scale = actual / max(rate.last_entropy_bytes + self._metadata_bytes(), 1.)
        rate.last_scale = scale
        return actual, scale

    def loss_terms(self, step, diagnostics=True):
        from .storage_rate import motion_rate_terms
        return motion_rate_terms(self, step, diagnostics=diagnostics)

    def _support_diagnostics(self, probabilities):
        region_t, region_r, gs_t, gs_r = self._values(hard=True)
        gt, gr = gs_t.ne(0).any(-1), gs_r.ne(0).any(-1)
        rt_active, rr_active = region_t.ne(0).any(-1), region_r.ne(0).any(-1)
        return dict(gs_active_rows=(gt | gr).sum(), gs_translation_active_blocks=gt.sum(),
            gs_rotation_active_blocks=gr.sum(), region_active_nodes=(rt_active | rr_active).sum(),
            spatial_active_regions=self.owner.unique().numel() * self.base_xyz.new_ones(()),
            spatial_open_gates=self.opened[1:].sum(), spatial_gate_mean_probability=(
                probabilities[self.split_eligible].mean() if self.split_eligible.any() else self.base_xyz.new_zeros(())))

    @torch.no_grad()
    def _record_predictor_proposal(self):
        """Save photo-only proposals before L1/L0 can erase branch evidence."""
        shared = self._shared_motion()
        target = torch.cat((shared[0] + self.gs_translation,
                            shared[1] + self.gs_rotation), -1).detach().clone()
        metrics = []
        for parameter, group in ((self.gs_translation, self.optimizer.param_groups[2]),
                                  (self.gs_rotation, self.optimizer.param_groups[3])):
            state = self.optimizer.state.get(parameter, {})
            if 'exp_avg_sq' not in state:
                metrics.append(torch.zeros_like(parameter))
                continue
            variance = state['exp_avg_sq'] / (1 - self.config.adam_betas[1] ** int(state['step']))
            metrics.append((variance.sqrt() + self.config.adam_eps) / group['lr'])
        self._predictor_fit_state = dict(target=target, metric=torch.cat(metrics, -1))

    def _predictor_rd_terms(self, step, diagnostics):
        from .storage_rate import predictor_rd_rate_terms
        return predictor_rd_rate_terms(self, step, diagnostics)

    @torch.no_grad()
    def _statistical_representatives(self, groups, symbols, observed):
        return self.regional_motion.statistical_representatives(self, groups, symbols, observed)

    @torch.no_grad()
    def _initialize_candidates(self, full_motion=None):
        return self.regional_motion.initialize_candidates(self, full_motion)

    @torch.no_grad()
    def _refresh_unselected_candidates(self):
        return self.regional_motion.refresh_unselected_candidates(self)

    @torch.no_grad()
    def _refresh_active_representatives(self):
        return self.regional_motion.refresh_active_representatives(self)

    @torch.no_grad()
    def _recode_owner(self, new_owner, full_motion=None, preserve_remainder=False):
        """Change the sharing representation without changing a rendered pose."""
        changed = new_owner.ne(self.owner)
        if full_motion is None and not changed.any():
            return
        if full_motion is None:
            shared_t, shared_r = self._shared_motion()
            total_t = torch.round(shared_t / self.steps[:3]) + torch.round(self.gs_translation / self.gs_steps[:3])
            total_r = torch.round(shared_r / self.steps[3:]) + torch.round(self.gs_rotation / self.gs_steps[3:])
        else:
            total_t, total_r = full_motion
            changed = torch.ones_like(changed)
        rt, rr = (value.float() for value in self._regional_predictions(hard=True))
        residual_t, residual_r = total_t[changed] - rt[new_owner[changed]], total_r[changed] - rr[new_owner[changed]]
        if residual_t.abs().max() > 32767 or residual_r.abs().max() > 32767:
            raise ValueError('Spatial recoding exceeds int16 residual range')
        if preserve_remainder:
            self.gs_translation[changed] = _retain_bin_remainder(
                residual_t, self.gs_translation[changed], self.gs_steps[:3])
            self.gs_rotation[changed] = _retain_bin_remainder(
                residual_r, self.gs_rotation[changed], self.gs_steps[3:])
        else:
            self.gs_translation[changed] = residual_t * self.gs_steps[:3]
            self.gs_rotation[changed] = residual_r * self.gs_steps[3:]
        # Additive recoding leaves the pose and its GS image Jacobian intact;
        # its optimizer history remains valid in the new relative coordinates.
        self.owner = new_owner

    @torch.no_grad()
    def _select_frontier(self, freeze=True, full_motion=None, preserve_remainder=False):
        if self.config.spatial_gate.frame_training_plan:
            before = tuple(value.clone() for value in self._full_motion_symbols())
            new_open, new_owner = self._region_gate_frontier()
            count = int(new_owner.ne(self.owner).sum())
            changes = torch.where(new_open.ne(self.opened))[0]
            if not self.shared_discovery_active and count:
                self._recode_owner(new_owner, preserve_remainder=True)
                after = self._full_motion_symbols()
                if not all(torch.equal(a, b) for a, b in zip(before, after)):
                    raise ValueError('Gate recoding changed quantized total motion')
            self.opened, self.owner = new_open, new_owner
            self._hard_gate_frozen = freeze
            self.flat_region_diagnostics.setdefault('reference_changes', []).append(
                dict(step=self.last_step, changed_gaussians=count,
                     preserved_total_motion=not self.shared_discovery_active))
            edits = [dict(kind='spatial_gate', node=int(n), opened=bool(new_open[n]), step=self.last_step)
                     for n in changes]
            self.topology_edits.extend(edits)
            return edits
        was_frozen = self._hard_gate_frozen
        if (not self._shared_field_active and full_motion is None and
                self._candidate_initialized and not self._coding_initialized and not self._hard_gate_frozen):
            full_motion = self._full_motion_symbols()
        self._coding_initialized = True
        new_open, new_owner = self._region_gate_frontier()
        gate_changes = torch.where(new_open != self.opened)[0]
        if self.shared_image_recode and self.joint_image_gradients:
            # Repartition the current fitted motion. Regional parameter
            # updates above remain physical; only a gate change rebases E.
            if not self.shared_discovery_active and bool(new_owner.ne(self.owner).any()):
                self._recode_owner(new_owner, preserve_remainder=True)
        elif self.joint_image_gradients:
            # The forward model and its STE must agree: a hard split changes
            # the selected shared field. Do not silently undo it in point
            # residuals; both fields continue fitting after the decision.
            pass
        elif self.joint_training and self._shared_field_active:
            # Switching representation must retain the pose just fitted by
            # the old frontier, even if gate logits crossed their threshold.
            if bool(new_owner.ne(self.owner).any()):
                rt, rr = (value.float() for value in self._regional_predictions(hard=True))
                full_motion = (rt[self.owner] + _integer(self.gs_translation, self.gs_steps[:3]),
                               rr[self.owner] + _integer(self.gs_rotation, self.gs_steps[3:]))
                self._recode_owner(new_owner, full_motion=full_motion, preserve_remainder=True)
        elif not self._shared_field_active:
            self._recode_owner(new_owner, full_motion=full_motion,
                               preserve_remainder=preserve_remainder)
        if freeze and not was_frozen and self.soft_image_gates:
            soft_t = self._shared_motion(relaxed=True)[0] + self._gs_motion(relaxed=True)[0]
            rt = self._regional_predictions()[0]
            gt = _quantized(self.gs_translation, self.gs_steps[:3])
            if self.private_full_motion:
                gt = gt - rt[self.spatial_paths[:, -1]]
            if self.hierarchical_private:
                gt = gt * new_owner.eq(self.spatial_paths[:, -1])[:, None]
            jump = (rt[new_owner] + gt - soft_t).norm(dim=-1)
            self.flat_region_diagnostics['gate_hardening'] = dict(step=self.last_step,
                mean_translation_jump=float(jump.mean()),
                p95_translation_jump=float(torch.quantile(jump, .95)),
                maximum_translation_jump=float(jump.max()),
                gs_quantization_step=float(self.gs_steps[0]))
        self._hard_gate_frozen = freeze
        edits = [dict(kind='spatial_gate', node=int(node), opened=bool(new_open[node]),
                      step=self.last_step) for node in gate_changes]
        self.opened, self.owner = new_open, new_owner
        if freeze and not was_frozen and self.config.spatial_gate.shared_discovery:
            factor = self._motion_lr_factor(self.last_step)
            self.optimizer.param_groups[2]['lr'] = self.config.gs_translation_lr * factor
            self.optimizer.param_groups[3]['lr'] = self.config.gs_rotation_lr * factor
            for parameter in (self.gs_translation, self.gs_rotation):
                state = self.optimizer.state.get(parameter, {})
                if 'exp_avg' in state:
                    # Keep sensitivity accumulated from real images, and
                    # fit a fresh direction under the newly fixed predictor.
                    state['exp_avg'].zero_()
        if freeze and not was_frozen and self.hierarchical_private:
            self._finalize_private_support()
            if self.config.spatial_gate.reparameterize_after_gate_freeze:
                self._begin_fixed_gate_residuals()
        elif freeze and not was_frozen and (self.soft_image_gates or self.stochastic_image_gates):
            # The last 30 steps fit the actual discrete field. Its physical
            # Jacobian differs from the soft mixture used during exploration.
            parameters = (self.region_translation, self.region_rotation)
            if not (self.shared_image_recode and self.joint_image_gradients):
                parameters += (self.gs_translation, self.gs_rotation)
            # Physical recoding uses the same GS image derivative before and
            # after freezing; only the auxiliary predictor objective stops.
            for parameter in parameters:
                self.optimizer.state.pop(parameter, None)
            if self.config.spatial_gate.reparameterize_after_gate_freeze:
                self._begin_fixed_gate_residuals()
        self.topology_edits.extend(edits)
        return edits

    @torch.no_grad()
    def _begin_shared_field_fit(self):
        full = self._full_motion_symbols()
        self._initialize_candidates(full_motion=full)
        for parameter in (self.gs_translation, self.gs_rotation):
            parameter.zero_()
            parameter.grad = None
            parameter.requires_grad_(False)
        self._shared_field_active = True
        self._coding_initialized = True
        return self._select_frontier(freeze=False)

    @torch.no_grad()
    def _begin_point_residual_fit(self):
        if not self._hard_gate_frozen or not self._shared_field_active:
            raise ValueError('Point residuals require the fitted shared field and selected tree')
        self._shared_field_active = False
        # Keep the learned shared field. Reintroducing absolute warm proposals
        # here would overwrite it and create compensating point residuals.
        for parameter in (self.gs_translation, self.gs_rotation):
            parameter.zero_()
            parameter.requires_grad_(True)
            self.optimizer.state.pop(parameter, None)
        self._enforce_rotation_policy()

    @torch.no_grad()
    def _proximal_relative_gs(self):
        return self.residual_motion._proximal_relative_gs(self)

    @torch.no_grad()
    def optimizer_step(self):
        cfg = self.config.spatial_gate
        self._mask_regional_fit_updates()
        if self.amplitude_refit_active:
            self._mask_amplitude_refit_updates()
        self._mask_region_gate_gradient()
        factor = self._motion_lr_factor(self.last_step)
        for index, (group, lr) in enumerate(zip(self.optimizer.param_groups, (self.config.region_translation_lr,
                self.config.region_rotation_lr, self.config.gs_translation_lr, self.config.gs_rotation_lr))):
            group['lr'] = 0. if self.shared_discovery_active and index >= 2 else lr * factor
        previous_shared = None
        if self.joint_training and not self.joint_image_gradients and not self.regional_image_updates:
            previous_shared = tuple(value.detach() for value in self._shared_motion())
        self.optimizer.step()
        for parameter, scale in ((self.region_translation, self.steps[:3]),
                (self.region_rotation, self.steps[3:]), (self.gs_translation, self.gs_steps[:3]),
                (self.gs_rotation, self.gs_steps[3:])):
            parameter.clamp_(min=-32767 * scale, max=32767 * scale)
        self._clamp_region_gate()
        if self.joint_training:
            # Coding-only coordinates compensate dictionary changes. Shared
            # image mode keeps the physical region update; only a hard gate
            # switch below rebases residuals without changing the fitted pose.
            if not self.joint_image_gradients and not self.regional_image_updates:
                full_motion = (torch.round(previous_shared[0] / self.steps[:3]) +
                               _integer(self.gs_translation, self.gs_steps[:3]),
                               torch.round(previous_shared[1] / self.steps[3:]) +
                               _integer(self.gs_rotation, self.gs_steps[3:]))
                self._rebase_full_motion(full_motion, preserve_remainder=True)
            if not self.amplitude_refit_active:
                if self.config.spatial_gate.predictor_objective == 'proximal_rd':
                    self._record_predictor_proposal()
                self._proximal_relative_gs()
            return self._finish_joint_step()
        initializing = self.last_step == cfg.start_step and not self._candidate_initialized
        edits = []
        if initializing:
            edits = self._begin_shared_field_fit()
        elif self._shared_field_active:
            if self.last_step >= cfg.gate_start_step:
                edits = self._select_frontier(freeze=self.last_step >= cfg.freeze_step)
                if self._hard_gate_frozen:
                    self._begin_point_residual_fit()
        elif self._candidate_initialized and self._motion_fit_window_due():
            if self._hard_gate_frozen and not self.joint_image_gradients:
                self.last_metrics['motion/refreshed_region_modes'] = self._refresh_active_representatives()
            self.last_metrics['motion/prox_zero_region_translation_blocks'] = 0
            self.last_metrics['motion/prox_zero_region_rotation_blocks'] = 0
            self._proximal_gs_blocks()
        if self.last_step in (cfg.start_step, cfg.gate_start_step, (cfg.start_step + cfg.freeze_step) // 2,
                              cfg.freeze_step, self.motion_steps):
            self._record_gate_learning()
        self.optimizer.zero_grad(set_to_none=True)
        return edits

    @torch.no_grad()
    def _finish_joint_step(self):
        cfg = self.config.spatial_gate
        edits = []
        if cfg.frame_training_plan:
            if self.last_step <= 100 or cfg.joint_gate_start_step <= self.last_step <= 180:
                edits = self._select_frontier(freeze=self.last_step in (100, 180))
            if self.last_step >= self.config.shared_discovery_steps:
                self._shared_field_active = False
            if self._motion_fit_window_due():
                self._proximal_gs_blocks()
            self.optimizer.zero_grad(set_to_none=True)
            return edits
        if (cfg.candidate_refresh_interval and not self._hard_gate_frozen and
                self.last_step < cfg.freeze_step and self.last_step % cfg.candidate_refresh_interval == 0):
            count = self._refresh_unselected_candidates()
            self.flat_region_diagnostics.setdefault('candidate_refresh_trace', []).append(
                dict(step=self.last_step, refreshed_nodes=count, representative='rate_relative_v1'))
        if not self._hard_gate_frozen:
            edits = self._select_frontier(freeze=self.last_step >= cfg.freeze_step)
            if self._hard_gate_frozen:
                # Keep point residuals and optimizer history across step 70.
                self._shared_field_active = False
                if cfg.predictor_objective == 'proximal_rd':
                    # The selected predictors now fit true image error. Their
                    # old moments belonged to a local proposal objective.
                    for parameter in (self.region_translation, self.region_rotation):
                        self.optimizer.state.pop(parameter, None)
        if self._motion_fit_window_due() and not self.amplitude_refit_active:
            if (self._hard_gate_frozen and not self.joint_image_gradients and
                    cfg.predictor_objective != 'proximal_rd'):
                self.last_metrics['motion/refreshed_region_modes'] = self._refresh_active_representatives()
            self._proximal_gs_blocks()
        if self.last_step in (1, cfg.freeze_step // 2, cfg.freeze_step, self.motion_steps):
            self._record_gate_learning()
        self.optimizer.zero_grad(set_to_none=True)
        return edits

    @torch.no_grad()
    def _proximal_gs_blocks(self):
        return self.residual_motion._proximal_gs_blocks(self)

    @torch.no_grad()
    def _record_gate_learning(self):
        selection = self._region_selection_probabilities()
        rt = _integer(self.region_translation, self.steps[:3]).ne(0).any(-1)
        rr = _integer(self.region_rotation, self.steps[3:]).ne(0).any(-1)
        layers, active = [], self.owner.unique()
        for level in range(self.spatial_paths.shape[1]):
            nodes = self.spatial_paths[:, level].unique()
            grad = self.region_translation.grad
            layers.append(dict(depth=level + 1, candidates=len(nodes),
                active_regions=int(torch.isin(active, nodes).sum()),
                nonzero_motion=int((rt[nodes] | rr[nodes]).sum()),
                mean_selection_probability=float(selection[nodes].mean()),
                translation_gradient_rms=(float(grad[nodes].square().mean().sqrt()) if grad is not None else 0.)))
        self.flat_region_diagnostics.setdefault('learning_trace', []).append(dict(
            step=self.last_step, hard_frozen=self._hard_gate_frozen,
            gate_temperature=float(self._gate_temperature()),
            image_relaxation=self.config.spatial_gate.image_relaxation,
            shared_field_active=self._shared_field_active,
            active_regions=len(active),
            active_regions_by_depth={str(layer['depth']): layer['active_regions'] for layer in layers},
            gate_gradient_norm=(float(self.gate_logits.grad.norm()) if self.gate_logits.grad is not None else 0.),
            coding_initialized=self._coding_initialized,
            top_gate_probabilities=self._gate_probabilities()[1:9].tolist(), layers=layers,
            private_allowed_gs=(int(self._private_support().sum()) if self.hierarchical_private else len(self.owner))))

    @torch.no_grad()
    def record_final_diagnostics(self):
        counts = torch.bincount(self.owner, minlength=len(self.parents))
        depth = {int(node): level + 1 for level in range(self.spatial_paths.shape[1])
                 for node in self.spatial_paths[:, level].unique().tolist()}
        active = torch.where(counts > 0)[0]
        # A small residual count alone does not establish correct motion. Keep
        # its distribution beside support and region size for image-error audits.
        gt, gr = (self._private_symbols() if self.private_full_motion else
                  (_integer(self.gs_translation, self.gs_steps[:3]), _integer(self.gs_rotation, self.gs_steps[3:])))
        residual = gt.ne(0).any(-1) | gr.ne(0).any(-1)
        residual_counts = torch.bincount(self.owner[residual], minlength=len(self.parents))
        observed_counts = torch.bincount(self.owner[self.observations.gt(0)],
                                        minlength=len(self.parents))
        leaf = self.spatial_paths[:, -1]
        leaf_counts = torch.bincount(leaf, minlength=len(self.parents))
        leaf_residual_counts = torch.bincount(leaf[residual], minlength=len(self.parents))
        # Materialize small diagnostic tables once. Reading CUDA scalars in
        # the region/gate loops otherwise synchronizes thousands of times.
        residual_fractions = (residual_counts / counts.clamp_min(1)).cpu()
        counts, residual_counts, observed_counts, leaf_counts, leaf_residual_counts = (
            table.cpu() for table in (counts, residual_counts, observed_counts,
                                     leaf_counts, leaf_residual_counts))
        active = active.cpu().tolist()
        translations = self.region_translation.cpu().tolist()
        rotations = self.region_rotation.cpu().tolist()
        gate_probabilities = self._gate_probabilities().cpu().tolist()
        gate_nodes = torch.where(self.split_eligible)[0].cpu().tolist()
        opened = self.opened.cpu()
        largest_leaf = int(leaf_counts.argmax())
        deepest = self.spatial_paths.shape[1]
        deepest_nodes = torch.tensor([node for node in active if depth[node] == deepest],
                                     dtype=torch.long)
        tree_health = dict(
            scope='surviving GS before this frame birth; counts do not measure image correctness',
            point_weighted_depth=sum(depth[int(node)] * int(counts[node]) for node in active) / len(self.owner),
            points_on_deepest_frontier=int(counts[deepest_nodes].sum()),
            residual_points_on_deepest_frontier=int(residual_counts[deepest_nodes].sum()),
            largest_candidate_leaf=dict(node=largest_leaf, count=int(leaf_counts[largest_leaf]),
                local_residual_count=int(leaf_residual_counts[largest_leaf])),
        )
        self.flat_region_diagnostics.update(
            semantics=('parent-relative region corrections' if self.relative_regions else 'absolute region increments') + '; orientation-only rotations; GS residuals',
            spatial_assignment='previous_decoded_position_v1',
            spatial_initialization=self.config.spatial_gate.initialization,
            spatial_reference_layout=self.config.spatial_gate.reference_layout,
            spatial_initialization_distance_quantile=self.config.spatial_gate.initialization_distance_quantile,
            spatial_outlier_assignment=self.config.spatial_gate.outlier_assignment,
            training_schedule=self.config.spatial_gate.training_schedule,
            region_representative=('previous region field; joint region and GS fit' if self.joint_training
                                   else 'quantized median initialization; shared field fit; point residual fit'),
            region_motion_learning=(self._regional_fit_support is not None and
                                    bool(self._regional_fit_support.any())),
            joint_image_gradients=self.joint_image_gradients,
            image_relaxation=self.config.spatial_gate.image_relaxation,
            hierarchical_private=self.hierarchical_private,
            proposal_image_weight=self.config.spatial_gate.proposal_image_weight,
            private_full_motion=self.private_full_motion,
            private_full_motion_configured=self.config.spatial_gate.private_full_motion,
            reparameterize_after_gate_freeze=self.config.spatial_gate.reparameterize_after_gate_freeze,
            predictor_objective=self.config.spatial_gate.predictor_objective,
            shared_image_recode=self.shared_image_recode,
            rotation_updates=self.config.rotation_updates,
            rate_model=self.config.rate_model,
            region_motion=self.config.spatial_gate.region_motion,
            stored_region_nodes=int(self._stored_region_support().sum()),
            candidate_nodes=len(self.parents), active_regions=len(active),
            open_gates=int(opened[1:].sum()), eligible_gates=len(gate_nodes),
            gate_start_step=self.config.spatial_gate.start_step,
            gate_learning_start_step=self.config.spatial_gate.gate_start_step,
            relative_motion_units='extent/radians; expected full-motion candidate residual',
            gate_freeze_step=self.config.spatial_gate.freeze_step,
            active_regions_by_depth={str(level): sum(depth[int(node)] == level for node in active)
                                     for level in range(1, self.spatial_paths.shape[1] + 1)},
            tree_health=tree_health,
            regions=[dict(node=int(node), depth=depth[int(node)], count=int(counts[node]),
                observed_count=int(observed_counts[node]), local_residual_count=int(residual_counts[node]),
                local_residual_fraction=float(residual_fractions[node]),
                translation=translations[node], rotation=rotations[node])
                for node in active],
            gates=[dict(node=node, probability=gate_probabilities[node],
                        opened=bool(opened[node])) for node in gate_nodes])

    @torch.no_grad()
    def decoded_survivor_pose(self):
        """CPU decoder-exact old pose without encoding or advancing state."""
        from .frame_stream_format import (compose_decoded_survivors)
        rt, rr, gt, gr = self._values(hard=True)
        return compose_decoded_survivors(self.frame_reference_state,self.extent,
            self.previous_rows,self.owner,self.parents,rt,rr,gt,gr,
            self.config.motion_xyz_step_extent,self.config.motion_rotation_step_rad,
            gs_translation_step_extent=self.config.gs_xyz_step_extent,
            gs_rotation_step_rad=self.config.gs_rotation_step_rad,
            spatial_open=self.opened,spatial_relative=self.relative_regions)

    def export_motion(self, path, inherit_birth_rotation=False, birth_pose_steps=None):
        from .frame_stream_encode import (encode_motion)
        from .frame_stream_decode import (decode_motion, motion_packet_info)
        if not self.committed:
            raise ValueError('Materialize spatial motion before exporting')
        rt, rr, gt, gr = [value.detach().cpu() for value in self._values(hard=True)]
        packet = encode_motion(self.frame_reference_state, self.frame_index, self.extent,
            self.previous_rows.cpu(), self.owner.cpu(), self.parents.cpu(), rt, rr, gt, gr,
            self.pending_birth_sources, self.materialized_xyz[len(self.owner):].cpu(),
            self.materialized_rotation[len(self.owner):].cpu(),
            self.config.motion_xyz_step_extent, self.config.motion_rotation_step_rad,
            compression='auto', gs_translation_step_extent=self.config.gs_xyz_step_extent,
            gs_rotation_step_rad=self.config.gs_rotation_step_rad, spatial_open=self.opened.cpu(),
            spatial_relative=self.relative_regions,
            inherit_birth_rotation=inherit_birth_rotation,birth_pose_steps=birth_pose_steps)
        decoded = decode_motion(packet, self.frame_reference_state)
        if not torch.allclose(decoded.xyz, self.materialized_xyz.cpu(), atol=1e-6, rtol=1e-5):
            raise ValueError('Spatial stream differs from the quantized training pose')
        dot = (F.normalize(decoded.rotation, dim=-1) * F.normalize(self.materialized_rotation.cpu(), dim=-1)).sum(-1).abs()
        if not bool((dot > 1 - 1e-5).all()):
            raise ValueError('Spatial stream differs from the quantized training orientation')
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(packet)
        self.codec_previous_state = decoded
        return dict(path=str(path), bytes=len(packet), payload_bytes=len(packet),
            **motion_packet_info(packet), frame_index=self.frame_index, gs_count=len(decoded.xyz),
            region_count=len(self.parents), active_region_count=int(self.owner.unique().numel()),
            region_translation_step_world=float(self.steps[0]), region_rotation_step_rad=float(self.steps[3]),
            gs_translation_step_world=float(self.gs_steps[0]), gs_rotation_step_rad=float(self.gs_steps[3]),
            semantics='spatial gates; ' + ('parent-relative region corrections' if self.relative_regions else 'absolute region increments') + '; no rotation of GS centers')

    def capture(self):
        packet = super().capture()
        if packet is not None:
            packet['representative_objective'] = 'rate_relative_v1'
            packet['gate_sparsification'] = 'fit_windows'
            packet['regional_image_fitting'] = ('stochastic_proposals_fixed_residuals_v1'
                                                if self.config.spatial_gate.reparameterize_after_gate_freeze else
                                                'joint_image_residual_stochastic_v1'
                                                if self.stochastic_image_gates and not self.config.spatial_gate.private_full_motion else
                                                'joint_image_private_proposal_stochastic_v1'
                                                if self.stochastic_image_gates else
                                                'joint_image_private_proposal_soft_v1'
                                                if self.private_full_motion and self.soft_image_gates else
                                                'joint_image_private_proposal_v1' if self.private_full_motion else
                                                'joint_image_soft_anneal_private_v1'
                                                if self.soft_image_gates and self.hierarchical_private else
                                                'joint_image_soft_anneal_v1' if self.soft_image_gates else
                                                'joint_image_hierarchical_private_v1' if self.hierarchical_private else
                                                'predictor_proposal_then_image_v1'
                                                if self.config.spatial_gate.predictor_objective == 'proximal_rd' else
                                                'shared_image_recode_v1' if self.shared_image_recode else
                                                'joint_image_dense_candidates_v2' if self.joint_image_gradients else
                                                'joint_motion_coordinates_v1' if self.joint_training
                                                else 'observed_candidates_v2')
            packet['candidate_initialization'] = ('previous_region_field_v1' if self.joint_training
                                                   else 'quantized_median_v1')
            packet['motion_schedule'] = ('joint_motion_v2' if self.joint_training
                                          else 'shared_field_zero_residual_v2')
            if self.joint_training:
                packet['gate_estimator'] = self._gate_estimator()
            if self.stochastic_image_gates:
                packet['image_gate_sampler'] = dict(rng_state=self._gate_rng.get_state().clone(),
                    step=self._sampled_gate_step,
                    noise=(None if self._sampled_gate_noise is None else self._sampled_gate_noise.detach().cpu().clone()))
            if self.config.spatial_gate.reparameterize_after_gate_freeze:
                packet['private_full_motion_active'] = self.private_full_motion
            packet['motion_initialization'] = ('previous_full_increment_v2' if self.joint_training
                                                else 'previous_increment_v1')
            packet['spatial_assignment'] = 'previous_decoded_position_v1'
            packet['shared_field_active'] = self._shared_field_active
            packet['regional_fit_support'] = self._regional_fit_support
            packet['parameters']['gate_logits'] = self.gate_logits.detach().clone()
            packet['tensors'].update({name: getattr(self, name) for name in
                ('spatial_paths', 'opened', 'node_counts', 'split_eligible')})
            packet['candidate_initialized'] = self._candidate_initialized
            packet['coding_initialized'] = self._coding_initialized
            packet['hard_gate_frozen'] = self._hard_gate_frozen
            packet['amplitude_refit_support'] = getattr(self, '_amplitude_refit_support', None)
            packet['codec_rate'] = (None if getattr(self, '_codec_rate', None) is None
                                    else self._codec_rate.capture())
            if self.config.spatial_gate.predictor_objective == 'proximal_rd':
                packet['predictor_fit_state'] = self._predictor_fit_state
        return packet

    def restore(self, packet):
        self._frame_hierarchical_private = self.config.spatial_gate.hierarchical_private
        self._frame_private_full_motion = self.config.spatial_gate.private_full_motion
        if packet is not None and packet.get('spatial_assignment') != 'previous_decoded_position_v1':
            raise ValueError('Fixed-membership spatial checkpoints are replay-only')
        initialization = 'previous_full_increment_v2' if self.joint_training else 'previous_increment_v1'
        schedule = 'joint_motion_v2' if self.joint_training else 'shared_field_zero_residual_v2'
        candidate = 'previous_region_field_v1' if self.joint_training else 'quantized_median_v1'
        if packet is not None and packet.get('motion_initialization') != initialization:
            raise ValueError('Earlier zero-initialized motion checkpoints are replay-only')
        if packet is not None and packet.get('motion_schedule') != schedule:
            raise ValueError('Earlier point-first motion schedules are replay-only')
        estimator = self._gate_estimator()
        if packet is not None and self.joint_training and packet.get('gate_estimator') != estimator:
            raise ValueError('Earlier joint gate estimators are replay-only')
        fitting = ('stochastic_proposals_fixed_residuals_v1'
                   if self.config.spatial_gate.reparameterize_after_gate_freeze else
                   'joint_image_residual_stochastic_v1'
                   if self.stochastic_image_gates and not self.config.spatial_gate.private_full_motion else
                   'joint_image_private_proposal_stochastic_v1' if self.stochastic_image_gates else
                   'joint_image_private_proposal_soft_v1' if self.private_full_motion and self.soft_image_gates else
                   'joint_image_private_proposal_v1' if self.private_full_motion else
                   'joint_image_soft_anneal_private_v1' if self.soft_image_gates and self.hierarchical_private else
                   'joint_image_soft_anneal_v1' if self.soft_image_gates else
                   'joint_image_hierarchical_private_v1' if self.hierarchical_private else
                   'predictor_proposal_then_image_v1'
                   if self.config.spatial_gate.predictor_objective == 'proximal_rd' else
                   'shared_image_recode_v1' if self.shared_image_recode else
                   'joint_image_dense_candidates_v2' if self.joint_image_gradients else
                   'joint_motion_coordinates_v1' if self.joint_training else 'observed_candidates_v2')
        if packet is not None and packet.get('regional_image_fitting') != fitting:
            raise ValueError('Earlier regional image-fitting schedules are replay-only')
        if packet is not None and packet.get('gate_sparsification') != 'fit_windows':
            raise ValueError('Earlier gate sparsification schedules are replay-only')
        if packet is not None and (packet.get('fixed_refit_support') is not None or
                packet['config'].get('post_birth_motion_steps', 0)):
            raise ValueError('Post-birth refinement checkpoints are replay-only')
        if packet is not None and packet.get('representative_objective') != 'rate_relative_v1':
            raise ValueError('Earlier region representative objectives are replay-only')
        if packet is not None and packet.get('candidate_initialization') != candidate:
            raise ValueError('Earlier candidate initializations are replay-only')
        super().restore(packet)
        if packet is not None:
            self._survivor_mapping_bytes = self._count_survivor_mapping_bytes()
        self._candidate_initialized = bool(packet and packet.get('candidate_initialized', False))
        self._shared_field_active = bool(packet and packet.get('shared_field_active', False))
        self._regional_fit_support = None if packet is None else packet['regional_fit_support']
        self._coding_initialized = bool(packet and packet.get('coding_initialized', packet.get('hard_gate_frozen', False)))
        self._hard_gate_frozen = bool(packet and packet.get('hard_gate_frozen', False))
        if packet is not None and self.config.spatial_gate.reparameterize_after_gate_freeze:
            active = packet.get('private_full_motion_active')
            if not isinstance(active, bool) or active == self._hard_gate_frozen:
                raise ValueError('Checkpoint private coordinates disagree with its gate stage')
            self._frame_private_full_motion = active
        self._amplitude_refit_support = None if packet is None else packet.get('amplitude_refit_support')
        self._predictor_fit_state = None if packet is None else packet.get('predictor_fit_state')
        if self._predictor_fit_state is not None:
            if (set(self._predictor_fit_state) != {'target', 'metric'} or any(
                    value.shape != (len(self.owner), 6) or not bool(torch.isfinite(value).all())
                    for value in self._predictor_fit_state.values())):
                raise ValueError('Invalid shared predictor fitting state')
            self._predictor_fit_state = {name: value.to(self.base_xyz).clone()
                                        for name, value in self._predictor_fit_state.items()}
        if self.amplitude_refit_active:
            if not self.config.amplitude_refit_steps or not self._hard_gate_frozen:
                raise ValueError('Invalid motion amplitude-refit checkpoint')
            expected = {'region_translation', 'region_rotation', 'gs_translation', 'gs_rotation'}
            if set(self._amplitude_refit_support) != expected:
                raise ValueError('Incomplete motion amplitude-refit support')
            for name, support in self._amplitude_refit_support.items():
                parameter = getattr(self, name)
                if support.dtype != torch.bool or support.shape != parameter.shape[:1]:
                    raise ValueError('Invalid motion amplitude-refit support shape')
                self._amplitude_refit_support[name] = support.to(parameter.device).clone()
        self._codec_rate = None
        if packet is not None and packet.get('codec_rate') is not None:
            from .storage_rate import MotionCodecRate
            self._codec_rate = MotionCodecRate.restore(packet['codec_rate'])
        if self._shared_field_active and not self.joint_training:
            for parameter in (self.gs_translation, self.gs_rotation):
                parameter.requires_grad_(False)
        if packet is not None:
            self._enforce_rotation_policy()
            if self.stochastic_image_gates:
                self._initialize_gate_sampler()
                sampler = packet.get('image_gate_sampler')
                if not isinstance(sampler, dict) or set(sampler) != {'rng_state', 'step', 'noise'}:
                    raise ValueError('Stochastic gate checkpoint lacks its sampler state')
                step, noise, rng = sampler['step'], sampler['noise'], sampler['rng_state']
                if (isinstance(step, bool) or not isinstance(step, int) or not 0 <= step <= self.last_step or
                        not torch.is_tensor(rng) or rng.dtype != torch.uint8 or rng.ndim != 1 or
                        (noise is None and step != 0) or
                        (noise is not None and (not torch.is_tensor(noise) or
                         noise.shape != self.gate_logits.shape or not noise.is_floating_point() or
                         not bool(torch.isfinite(noise).all())))):
                    raise ValueError('Invalid stochastic gate sampler state')
                try:
                    self._gate_rng.set_state(rng.cpu())
                except RuntimeError as error:
                    raise ValueError('Invalid stochastic gate RNG state') from error
                self._sampled_gate_step = step
                self._sampled_gate_noise = None if noise is None else noise.to(self.gate_logits).clone()


class FullUpdateSpatialMotionTraining(SpatialMotionTraining):
    """Default motion behavior unless the current frame requests full updates."""
    regional_repair = False
    original_rotation_updates = False
    shared_region_orientation = False
    color_only = False
    regional_full_200 = False

    @torch.no_grad()
    def _enforce_rotation_policy(self):
        if self.regional_repair and not self.original_rotation_updates:
            # Optionally fit only shared region orientation in motion fitting;
            # independent point rotations remain reserved for selected repair.
            if self.last_step <= 200:
                frozen = ((self.gs_rotation,) if self.shared_region_orientation else
                          (self.region_rotation, self.gs_rotation))
                for parameter in frozen:
                    parameter.zero_()
                    parameter.grad = None
                    parameter.requires_grad_(False)
                    self.optimizer.state.pop(parameter, None)
            return
        super()._enforce_rotation_policy()

    @torch.no_grad()
    def prepare_plan_step(self, step):
        if self.regional_repair and step > 200:
            self.last_step = int(step)
            self._hard_gate_frozen = True
            for parameter in (self.gate_logits, self.region_translation, self.region_rotation):
                parameter.requires_grad_(False)
                parameter.grad = None
            for parameter in (self.gs_translation, self.gs_rotation):
                parameter.requires_grad_(not self.color_only)
                if self.color_only:
                    parameter.grad = None
            return
        super().prepare_plan_step(step)
        if self.regional_full_200 and step <= 100:
            for parameter in (self.region_rotation, self.gs_rotation):
                parameter.requires_grad_(False)
                parameter.grad = None

    @torch.no_grad()
    def optimizer_step(self):
        if self.regional_repair:
            if self.last_step <= 200:
                return super().optimizer_step()
            self.optimizer.step()
            if self.color_only:
                self.optimizer.zero_grad(set_to_none=True)
                return []
            for parameter, grid in ((self.gs_translation, self.gs_steps[:3]),
                                    (self.gs_rotation, self.gs_steps[3:])):
                parameter.clamp_(min=-32767 * grid, max=32767 * grid)
            self._proximal_relative_gs()
            if self._motion_fit_window_due() or self.last_step == 300:
                self._proximal_gs_blocks()
            self.optimizer.zero_grad(set_to_none=True)
            return []
        return super().optimizer_step()
