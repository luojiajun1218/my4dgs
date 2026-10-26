"""Source-anchored detail training, with common material kept as an option.

Stage-zero common colour C is saved separately. Fine training always starts
mother and details at immutable source material theta0, so independently learned
child innovation D is measured in the same background used by its commit
endpoint. The controller retains the V10 own-view branches and codec checks.
"""

import torch

from .birth_quantization import RAW_FIELDS, RAW_ATTRIBUTES
from .conservative_family import _clone_tree
from .predictive_reincarnation_model import PredictiveReincarnationModel
from .reincarnation_model import project_sh1_sphere


SOURCE_RESIDUAL_SCOPE = ('Fine P material: immutable source prior; whole-family common C; '
    'independently learned child D in a source-colour mother background; bounded C+D. '
    'N material stays at its current mature value in every endpoint.')


class SourceResidualModel(PredictiveReincarnationModel):
    """V11 small extension; the frozen V10 model and format remain unchanged."""

    def __init__(self, sh_degree):
        super().__init__(sh_degree)
        self.source_residual_fine = False
        self.common_material = None

    def endpoint_kinds(self):
        return (('prior', 'common', 'detail', 'full') if self.source_residual_fine else
                ('prior', 'dc', 'sh', 'full'))

    def install_reincarnation_state(self, state):
        self.source_residual_fine = bool(state.get('source_residual_fine', False))
        super().install_reincarnation_state(state)
        self.common_material = _clone_tree(state.get('common_material'), self._xyz.device)
        if self.common_material is not None:
            for name, shape in (('sh_dc', self.p_common_dc.shape), ('sh_rest', self.p_common_sh1.shape)):
                value = self.common_material[name]
                if value.shape != shape or not bool(torch.isfinite(value).all()):
                    raise ValueError('Saved common material must align with immutable source roots')
        return self

    def reincarnation_state(self, device='cpu'):
        state = super().reincarnation_state(device)
        state.update(source_residual_fine=self.source_residual_fine,
            common_material=_clone_tree(self.common_material, device),
            source_residual_version='source_residual_support_v11',
            endpoint_scope=SOURCE_RESIDUAL_SCOPE)
        return state

    @torch.no_grad()
    def project_parameters(self):
        # A source on its own HDR envelope can land one ULP outside the cone
        # after midpoint/radius arithmetic. Its native coefficients define that
        # envelope and must remain exact; this exception is only for an actual
        # pristine point, never a cancellation or a learned common/detail value.
        prior_dc, prior_rest = self._prior_features()
        dc, rest = self._features_dc.clone(), self._features_rest.clone()
        dc[self.reincarnation_rows] = self.p_common_dc
        rest[self.reincarnation_rows] = self.p_common_sh1
        pristine = ((dc == prior_dc).all((1, 2)) & (rest == prior_rest).all((1, 2)))
        super().project_parameters()
        self._features_dc[pristine] = prior_dc[pristine]
        self._features_rest[pristine] = prior_rest[pristine]
        self.p_common_dc.copy_(self._features_dc[self.reincarnation_rows])
        self.p_common_sh1.copy_(self._features_rest[self.reincarnation_rows])
        self._clear_reincarnation_cache()
        return self

    project_ = project_parameters

    @torch.no_grad()
    def install_details_from_coarse(self, root_indices=None, detail_alpha=.1):
        material = (dict(sh_dc=self.p_common_dc.detach().clone(), sh_rest=self.p_common_sh1.detach().clone())
                    if self.common_material is None else _clone_tree(self.common_material))
        result = super().install_details_from_coarse(root_indices, detail_alpha)
        result.common_material = material
        result.source_residual_fine = True
        # Reset every P row, including roots that did not receive details.
        # N latent/initial state is unaffected. Common C remains an independent
        # immutable option rather than a background for learning child D.
        n = result.novel_count
        dc, rest = result._prior_features()
        result._features_dc[n:].copy_(dc[n:])
        result._features_rest[n:].copy_(rest[n:])
        result.p_common_dc.copy_(result.support_anchor_raw['sh_dc'])
        result.p_common_sh1.copy_(result.support_anchor_raw['sh_rest'])
        result._predictive_endpoints = None
        result.predictive_weights = None
        result.project_parameters()
        result.set_training_stage(1)
        return result

    @torch.no_grad()
    def freeze_predictive_endpoints(self, force=False):
        if not self.source_residual_fine:
            return super().freeze_predictive_endpoints(force)
        if self._predictive_endpoints is not None and not force:
            return self
        if self.common_material is None:
            raise ValueError('Fine source residual model lost its independent common material')
        learned = _clone_tree(self.reincarnation_phenotype())
        prior = _clone_tree(learned)
        n = self.novel_count
        prior_dc, prior_rest = self._prior_features()
        prior['sh_dc'][n:], prior['sh_rest'][n:] = prior_dc[n:], prior_rest[n:]
        common = _clone_tree(prior)
        root = self._point_root
        common['sh_dc'][n:] = self.common_material['sh_dc'][root]
        common['sh_rest'][n:] = self.common_material['sh_rest'][root]
        detail = _clone_tree(learned)
        # Mother has no D. At this point common C has never been applied during
        # fine SGD; explicitly keep its prior in the diagnostic detail endpoint.
        for name in ('sh_dc', 'sh_rest'):
            detail[name][self.reincarnation_rows] = prior[name][self.reincarnation_rows]
        full = _clone_tree(learned)
        for name in ('sh_dc', 'sh_rest'):
            full[name][n:] = common[name][n:]+(detail[name][n:]-prior[name][n:])
        low, high = self._radiance_bounds()
        dc, rest, _ = project_sh1_sphere(full['sh_dc'], full['sh_rest'], low, high)
        full['sh_dc'][n:], full['sh_rest'][n:] = dc[n:], rest[n:]
        pristine = (((common['sh_dc'] == prior['sh_dc']).all((1, 2))) &
                    ((common['sh_rest'] == prior['sh_rest']).all((1, 2))) &
                    ((detail['sh_dc'] == prior['sh_dc']).all((1, 2))) &
                    ((detail['sh_rest'] == prior['sh_rest']).all((1, 2))))
        for name in ('sh_dc', 'sh_rest'):
            full[name] = torch.where(pristine[:, None, None], prior[name], full[name])
        self._predictive_endpoints = dict(prior=prior, common=common, detail=detail, full=full)
        self.predictive_weights = None
        return self

    def endpoint_raw(self, kind):
        if not self.source_residual_fine:
            return super().endpoint_raw(kind)
        kind = {'common_only':'common', 'detail_only':'detail'}.get(kind, kind)
        if kind not in self.endpoint_kinds():
            raise ValueError('Fine source residual endpoint must be prior, common, detail or full')
        self.freeze_predictive_endpoints()
        return _clone_tree(self._predictive_endpoints[kind])

    @torch.no_grad()
    def apply_predictive_weights(self, weights):
        if not self.source_residual_fine:
            return super().apply_predictive_weights(weights)
        self.freeze_predictive_endpoints()
        w = torch.as_tensor(weights, device=self._xyz.device, dtype=torch.float64).flatten()
        if (w.shape != (3,) or not bool(torch.isfinite(w).all()) or
                bool((w < -1.e-10).any()) or float(w.sum()) > 1+1.e-10):
            raise ValueError('Three predictive weights must be nonnegative with sum <=1')
        w = w.clamp_min(0)
        if float(w.sum()) > 1:
            w = w/w.sum()
        prior = self._predictive_endpoints['prior']
        for name, attr in (('sh_dc', '_features_dc'), ('sh_rest', '_features_rest')):
            value = prior[name].clone()
            for weight, kind in zip(w.to(value), self.endpoint_kinds()[1:]):
                value.add_(self._predictive_endpoints[kind][name]-prior[name], alpha=float(weight))
            value[:self.novel_count] = prior[name][:self.novel_count]
            getattr(self, attr).copy_(value)
        self.p_common_dc.copy_(self._features_dc[self.reincarnation_rows])
        self.p_common_sh1.copy_(self._features_rest[self.reincarnation_rows])
        self.predictive_weights = w.detach().clone()
        # Do not reset the mother here: a validated common/full commit is
        # allowed to change it. Fine training froze its leaf, not its phenotype.
        self.project_parameters()
        return self

    def reincarnation_subset(self, selection):
        material = self.common_material
        self.common_material = None
        try:
            result = super().reincarnation_subset(selection)
        finally:
            self.common_material = material
        if material is not None:
            choice = torch.as_tensor(selection, device=self._xyz.device)
            ids = choice.nonzero().flatten() if choice.dtype == torch.bool else choice.long().flatten()
            roots = torch.tensor([i for i, rows in enumerate(self.reincarnation_groups)
                                  if bool(torch.isin(rows, ids).all())], device=self._xyz.device, dtype=torch.long)
            result.common_material = {name:value[roots].detach().clone() for name,value in material.items()}
        return result

    def capture(self):
        packet = super().capture()
        packet['source_residual'] = packet.pop('predictive_reincarnation')
        return packet

    def restore(self, packet, training_args, device=None):
        packet = dict(packet)
        packet['predictive_reincarnation'] = packet['source_residual']
        return super().restore(packet, training_args, device)
