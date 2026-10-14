"""Past-decoded translation initialization, separate from current evidence."""

from copy import deepcopy

import torch

from .causal_translation_prediction import propose_causal_translation_initialization


class CausalTranslationInitializationMixin:
    def _reset_causal_translation_prediction(self):
        self.causal_predicted_ids = torch.empty(0, dtype=torch.long)
        self.causal_initial_translation = torch.empty((0, 3), dtype=torch.float32)
        self.causal_prediction_metrics = {}
        self.causal_prediction_frame = -1
        self.causal_prediction_fallback_completed = False

    @torch.no_grad()
    def _initialize_causal_translation_prediction(self):
        self._reset_causal_translation_prediction()
        if not self.config.causal_translation_init:
            return
        proposal = propose_causal_translation_initialization(self.online_history,
            self.online_ids.numpy(), self.frame_index, self.gs_steps[:3].cpu().numpy(),
            self.extent, self.config.online_clustering,
            gs_xyz_step_extent=(self.config.gs_xyz_step_extent
                if self.config.gs_xyz_step_extent is not None else self.config.motion_xyz_step_extent))
        translation = torch.from_numpy(proposal.translation)
        selected = torch.from_numpy(proposal.predicted_mask).nonzero(as_tuple=True)[0]
        order = torch.argsort(self.online_ids[selected])
        selected = selected[order]
        self.causal_predicted_ids = self.online_ids[selected].clone()
        if not torch.equal(self.causal_predicted_ids, torch.from_numpy(proposal.pending_predicted_ids)):
            raise ValueError('Predicted stable IDs and initial values disagree')
        self.causal_initial_translation = translation[selected].clone()
        self.gs_translation.copy_(translation.to(self.gs_translation))
        self.causal_prediction_frame = self.frame_index
        self.causal_prediction_metrics = dict(proposal.metrics,
            applied_frame=self.frame_index, fallback_completed=False,
            current_observations_added=0, current_history_transitions_added=0,
            initialization_scope='Independent GS translation only; regional and rotation values remain zero',
            corrections={})

    def _causal_prediction_rows(self):
        identifiers = self.causal_predicted_ids
        if not len(identifiers):
            return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)
        order = torch.argsort(self.online_ids)
        ordered = self.online_ids[order]
        locations = torch.searchsorted(ordered, identifiers)
        matched = locations < len(ordered)
        matched[matched.clone()] &= ordered[locations[matched]] == identifiers[matched]
        selected = matched.nonzero(as_tuple=True)[0]
        return order[locations[matched]], selected

    @torch.no_grad()
    def _fallback_unobserved_translation_prediction(self):
        if (not self.config.causal_translation_init or self.causal_prediction_fallback_completed):
            return
        if self.last_step != self.config.motion_steps:
            raise ValueError('Prediction fallback belongs immediately before partition')
        rows, _ = self._causal_prediction_rows()
        rows = rows.to(self.gs_translation.device)
        unseen = rows[self.observations[rows] < self.config.online_clustering.min_observations]
        self.gs_translation[unseen] = 0
        if self.gs_translation.grad is not None:
            self.gs_translation.grad[unseen] = 0
        state = self.optimizer.state.get(self.gs_translation, {})
        for name in ('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'):
            value = state.get(name)
            if value is not None:
                value[unseen] = 0
        self.causal_prediction_fallback_completed = True
        self.causal_prediction_metrics.update(fallback_completed=True,
            fallback_step=self.last_step, unobserved_fallback_rows=len(unseen),
            observed_prediction_support=len(rows) - len(unseen),
            fallback_scope='Current radii-proxy observations only; clear row moments, preserve Adam scalar step')

    @torch.no_grad()
    def _record_translation_prediction_correction(self, step):
        if not self.config.causal_translation_init:
            return
        rows, initial_rows = self._causal_prediction_rows()
        rows = rows.to(self.base_xyz.device)
        observed = self.observations[rows] >= self.config.online_clustering.min_observations
        current_xyz, _ = self.materialized_pose(step)
        fitted = current_xyz[rows] - self.base_xyz[rows]
        initial = self.causal_initial_translation[initial_rows].to(fitted)
        correction = (fitted[observed] - initial[observed]).norm(dim=-1).cpu()
        record = dict(step=int(step), predicted_rows=len(self.causal_predicted_ids),
            matched_rows=len(rows), currently_observed_rows=int(observed.sum()),
            unobserved_fallback_rows=self.causal_prediction_metrics.get('unobserved_fallback_rows', 0),
            correction_scope='Model-fitted full WORLD displacement minus initial prediction, not GT flow error',
            observation_scope='Accumulated current-frame radii>0 proxy, not unoccluded pixel support',
            correction_mean=float(correction.mean()) if len(correction) else None,
            correction_p50=float(torch.quantile(correction, .5)) if len(correction) else None,
            correction_p90=float(torch.quantile(correction, .9)) if len(correction) else None)
        self.causal_prediction_metrics.setdefault('corrections', {})[str(step)] = record
        self.flat_region_diagnostics['translation_initialization'] = deepcopy(self.causal_prediction_metrics)

    def _capture_causal_translation_prediction(self):
        return dict(predicted_ids=self.causal_predicted_ids.clone(),
            initial_translation=self.causal_initial_translation.clone(),
            metrics=deepcopy(self.causal_prediction_metrics), frame=self.causal_prediction_frame,
            fallback_completed=self.causal_prediction_fallback_completed)

    def _restore_causal_translation_prediction(self, packet):
        self._reset_causal_translation_prediction()
        if packet is None:
            if self.config.causal_translation_init:
                raise ValueError('Enabled causal initialization checkpoint lost pending prediction state')
            return
        identifiers = torch.as_tensor(packet['predicted_ids']).detach().cpu()
        initial = torch.as_tensor(packet['initial_translation']).detach().cpu()
        if (identifiers.dtype != torch.long or identifiers.ndim != 1 or
                initial.dtype != torch.float32 or initial.shape != (len(identifiers), 3) or
                not bool(torch.isfinite(initial).all()) or bool(identifiers.lt(0).any()) or
                (len(identifiers) > 1 and not bool(identifiers[1:].gt(identifiers[:-1]).all()))):
            raise ValueError('Invalid causal translation prediction checkpoint')
        if not self.config.causal_translation_init:
            if len(identifiers):
                raise ValueError('Disabled initializer checkpoint contains predictions')
            return
        frame, completed, metrics = packet.get('frame'), packet.get('fallback_completed'), packet.get('metrics')
        if type(frame) is not int or frame != self.frame_index:
            raise ValueError('Causal prediction checkpoint frame differs from the motion frame')
        if type(completed) is not bool or completed != (self.last_step >= self.config.motion_steps):
            raise ValueError('Causal prediction fallback flag differs from the saved training phase')
        if (not isinstance(metrics, dict) or metrics.get('current_frame') != frame or
                metrics.get('applied_frame') != frame or metrics.get('predicted_gs') != len(identifiers) or
                type(metrics.get('fallback_completed')) is not bool or
                metrics['fallback_completed'] != completed):
            raise ValueError('Causal prediction checkpoint metrics disagree with its state')
        if not bool(torch.isin(identifiers, self.online_ids).all()):
            raise ValueError('Causal predicted ID is absent from current stable GS IDs')
        self.causal_predicted_ids = identifiers.clone()
        self.causal_initial_translation = initial.clone()
        self.causal_prediction_metrics = deepcopy(metrics)
        self.causal_prediction_frame = frame
        self.causal_prediction_fallback_completed = completed
