"""P-only source texture births for one 100-step birth stage.

No common-material fit and no new angular material: source SH1 is physically
fixed, while child DC, support and alpha can learn. This also makes a native
2x2 DC initializer genuinely DC-only, independent of the codec predictor grid.
"""

import torch

from .conservative_family import _clone_tree
from .reincarnation_model import C0, _sh1_vector, _ste
from .source_residual_model import SourceResidualModel


class QuickSourceResidualModel(SourceResidualModel):
    def install_reincarnations(self, specs, novel_raw=None, views=None):
        if novel_raw is not None and len(novel_raw['xyz']):
            raise ValueError('Quick source residual births are P-only')
        return super().install_reincarnations(specs, novel_raw, views)

    def set_training_stage(self, phase):
        super().set_training_stage(phase)
        # There is no stage-zero common fit in this lifecycle. Every existing
        # SH1 coefficient remains its source coefficient in all later phases.
        for parameter in (self._features_rest, self.p_common_dc, self.p_common_sh1):
            parameter.requires_grad_(False)
            parameter.grad = None
        return self

    def reincarnation_state(self, device='cpu'):
        state = super().reincarnation_state(device)
        state.update(quick_source_residual=True,
            radiance_scope='Single all-training-view P pool; fixed physical source SH1; child DC innovation; no common fit',
            endpoint_scope='Internal compatibility only; quick commit uses trained source-family codec increments')
        return state

    def intrinsic_dc_bounds(self):
        """Absolute DC bounds at the fixed physical SH1, aligned with all rows."""
        prior_dc, rest = self._prior_features()
        low, high = self._radiance_bounds()
        norm = torch.linalg.vector_norm(_sh1_vector(rest), dim=-1)
        lower = ((low+norm-.5)/C0)[:, None, :]
        upper = ((high-norm-.5)/C0)[:, None, :]
        # Envelope arithmetic can move a native boundary by an ULP. The
        # material that defines the envelope is always a feasible endpoint.
        return torch.minimum(lower, prior_dc), torch.maximum(upper, prior_dc)

    def _bounded_features(self, dc, rest, config=None, predictor=None):
        prior_dc, physical_rest = self._prior_features()
        original = dc
        target = dc
        if config is not None:
            step = dc.new_tensor(config.new_sh_step)
            target = predictor['sh_dc']+((dc-predictor['sh_dc'])/step).round()*step
        lower, upper = self.intrinsic_dc_bounds()
        target = torch.maximum(lower, torch.minimum(target, upper))
        target = torch.where(original.detach() == prior_dc, prior_dc, target)
        return _ste(original, target), physical_rest.detach().clone()

    @torch.no_grad()
    def project_parameters(self):
        dc = self._features_dc.clone()
        dc[self.reincarnation_rows] = self.p_common_dc
        dc, rest = self._bounded_features(dc, self._features_rest)
        self._features_dc.copy_(dc)
        self._features_rest.copy_(rest)
        self.p_common_dc.copy_(dc[self.reincarnation_rows])
        self.p_common_sh1.copy_(rest[self.reincarnation_rows])
        if self.predictive_phase == 1:
            rows, floor = self._maturity_opacity_floor()
            self._opacity[rows] = torch.maximum(self._opacity[rows], floor)
        self._clear_reincarnation_cache()
        return self

    project_ = project_parameters

    @torch.no_grad()
    def install_details_from_coarse(self, root_indices=None, detail_alpha=.1):
        result = super().install_details_from_coarse(root_indices, detail_alpha)
        result.common_material = dict(sh_dc=result.support_anchor_raw['sh_dc'].clone(),
                                      sh_rest=result.support_anchor_raw['sh_rest'].clone())
        result._predictive_endpoints = None
        result.predictive_weights = None
        return result

    @torch.no_grad()
    def freeze_predictive_endpoints(self, force=False):
        if not self.source_residual_fine:
            return super().freeze_predictive_endpoints(force)
        if self._predictive_endpoints is not None and not force:
            return self
        detail = _clone_tree(self.reincarnation_phenotype())
        prior = _clone_tree(detail)
        dc, rest = self._prior_features()
        prior['sh_dc'], prior['sh_rest'] = dc.clone(), rest.clone()
        detail['sh_dc'][self.reincarnation_rows] = dc[self.reincarnation_rows]
        self._predictive_endpoints = dict(prior=prior, common=_clone_tree(prior),
                                         detail=detail, full=_clone_tree(detail))
        self.predictive_weights = None
        return self

    def capture(self):
        packet = super().capture()
        packet['quick_source_residual'] = packet.pop('source_residual')
        return packet

    def restore(self, packet, training_args, device=None):
        packet = dict(packet)
        packet['source_residual'] = packet['quick_source_residual']
        return super().restore(packet, training_args, device)
