"""One Quick population with transparent, not-yet-accepted source families.

Only the accepted prefix and one explicitly probed family enter the scene.
The controller uses the entity mask to retire their original parents. A probe
never changes latent opacity or the accepted prefix; an accepted data density
becomes the family's actual alpha and maturity floor, with ordinary GS output.
"""
import math

import torch

from .birth_quantization import RAW_FIELDS
from .conservative_family import _clone_tree
from .predictive_reincarnation_model import _zero_opacity
from .quick_source_residual_model import QuickSourceResidualModel


class ShadowSourceResidualModel(QuickSourceResidualModel):
    def __init__(self, sh_degree):
        super().__init__(sh_degree)
        self.shadow_accepted = torch.empty(0, dtype=torch.bool)
        self.shadow_rho = torch.empty(0, dtype=torch.float64)
        self.shadow_probe_root = None
        self.shadow_probe_rho = 0.

    def _root_ids(self):
        return torch.tensor([int(spec['old_row']) for spec in self.reincarnation_specs],
                            device=self._xyz.device, dtype=torch.long)

    def shadow_state(self, device='cpu'):
        return _clone_tree(dict(root_ids=self._root_ids(), accepted=self.shadow_accepted,
            rho=self.shadow_rho, probe_root=self.shadow_probe_root,
            probe_rho=self.shadow_probe_rho), device)

    def install_shadow_state(self, state=None):
        """Restore metadata only; retain the captured latent/learned opacity.

        Immutable old-row IDs remap state BEFORE a V9-derived subset installer
        needs it, so accepted masks also work on an encoded candidate subset.
        """
        ids=self._root_ids();count=len(ids);device=self._xyz.device
        if state is None:
            accepted=torch.zeros(count,device=device,dtype=torch.bool)
            rho=torch.zeros(count,device=device,dtype=torch.float64);probe=None;probe_rho=0.
        else:
            old_ids=torch.as_tensor(state['root_ids'],dtype=torch.long).tolist()
            lookup={int(row):i for i,row in enumerate(old_ids)}
            if len(lookup)!=len(old_ids) or any(int(row) not in lookup for row in ids.tolist()):
                raise ValueError('Shadow state must map unique immutable source roots')
            selection=torch.tensor([lookup[int(row)] for row in ids.tolist()],device=device,dtype=torch.long)
            old_accepted=torch.as_tensor(state['accepted'],device=device,dtype=torch.bool)
            old_rho=torch.as_tensor(state['rho'],device=device,dtype=torch.float64)
            if old_accepted.shape!=(len(old_ids),) or old_rho.shape!=old_accepted.shape:
                raise ValueError('Shadow acceptance and density arrays must align with source IDs')
            accepted=old_accepted[selection].clone();rho=old_rho[selection].clone()
            if not bool(torch.isfinite(rho).all()) or bool(((rho<0)|(rho>1)).any()) or bool((accepted&(rho<=0)).any()):
                raise ValueError('Accepted shadow densities must be finite and in (0,1]')
            old_probe=state.get('probe_root');probe=None;probe_rho=0.
            if old_probe is not None:
                if not 0<=int(old_probe)<len(old_ids):
                    raise ValueError('Shadow probe root is outside its saved source population')
                probe_id=old_ids[int(old_probe)]
                current={int(row):i for i,row in enumerate(ids.tolist())}
                if probe_id in current:
                    probe=current[probe_id];probe_rho=float(state.get('probe_rho',0.))
                    if not math.isfinite(probe_rho) or not 0<=probe_rho<=1:
                        raise ValueError('Shadow probe density must be in [0,1]')
        self.shadow_accepted,self.shadow_rho=accepted,rho
        self.shadow_probe_root,self.shadow_probe_rho=probe,probe_rho
        for root,spec in enumerate(self.reincarnation_specs):
            spec['detail_birth_alpha']=.99*float(rho[root]) if bool(accepted[root]) else 0.
        self._clear_reincarnation_cache()
        return self

    def install_reincarnation_state(self, state):
        super().install_reincarnation_state(state)
        return self.install_shadow_state(state.get('shadow'))

    def reincarnation_state(self, device='cpu'):
        state=super().reincarnation_state(device)
        state.update(shadow_source_residual=True,shadow=self.shadow_state(device),
            radiance_scope='Single accepted source-family prefix; one transient density probe; fixed physical source SH1',
            endpoint_scope='No endpoint selection; ordinary GS attributes encode accepted data densities')
        return state

    def _check_pair(self, root):
        root=int(root)
        if not 0<=root<len(self.reincarnation_groups) or len(self.reincarnation_groups[root])!=3:
            raise ValueError('A shadow candidate must be one complete mother/two-child family')
        return root

    def _bounded_features(self, dc, rest, config=None, predictor=None):
        # A later accepted family can retire the nearest codec predictor of an
        # earlier one. Its density polynomial requires that prefix's physical
        # material to remain fixed across the rebase. The existing exact ATTR
        # fallback pays for off-grid DC, just as for fixed physical SH1/pose.
        return super()._bounded_features(dc,rest,config=None,predictor=None)

    def shadow_entity_mask(self):
        result=torch.zeros(len(self._xyz),device=self._xyz.device,dtype=torch.bool)
        for root,rows in enumerate(self.reincarnation_groups):
            if bool(self.shadow_accepted[root]) or (root==self.shadow_probe_root and self.shadow_probe_rho>0):
                result[rows]=True
        return result

    def set_shadow_probe(self, root, rho):
        root=self._check_pair(root);rho=float(rho)
        if bool(self.shadow_accepted[root]):
            raise ValueError('A probe cannot replace an already accepted source family')
        if not math.isfinite(rho) or not 0<=rho<=1:
            raise ValueError('Shadow probe density must lie in [0,1]')
        self.shadow_probe_root,self.shadow_probe_rho=root,rho
        self._clear_reincarnation_cache()
        return self

    def clear_shadow_probe(self):
        self.shadow_probe_root,self.shadow_probe_rho=None,0.
        self._clear_reincarnation_cache()
        return self

    @torch.no_grad()
    def accept_shadow_family(self, root, rho):
        root=self._check_pair(root);rho=float(rho)
        if not math.isfinite(rho) or not 0<rho<=1:
            raise ValueError('Accepted shadow density must lie in (0,1]')
        alpha=.99*rho;rows=self.reincarnation_groups[root]
        value=self._opacity.new_full((2,1),alpha)
        self._opacity[rows[1:]]=value.log()-torch.log1p(-value)
        self.shadow_accepted[root]=True;self.shadow_rho[root]=rho
        self.reincarnation_specs[root]['detail_birth_alpha']=alpha
        self.clear_shadow_probe()
        return self

    @torch.no_grad()
    def reject_shadow_family(self, root):
        root=self._check_pair(root)
        self.shadow_accepted[root]=False;self.shadow_rho[root]=0
        self.reincarnation_specs[root]['detail_birth_alpha']=0.
        rows=self.reincarnation_groups[root]
        self._opacity[rows[1:]]=_zero_opacity(self._opacity[rows[1:]])
        self.clear_shadow_probe()
        return self

    def _phenotype(self, config=None, extent=None, reference=None, source_rows=None):
        result=super()._phenotype(config,extent,reference,source_rows)
        mask=self.shadow_entity_mask()
        result['opacity']=torch.where(mask[:,None],result['opacity'],_zero_opacity(result['opacity']))
        root=self.shadow_probe_root
        if root is not None and self.shadow_probe_rho>0:
            rows=self.reincarnation_groups[root]
            alpha=result['opacity'].new_full((2,1),.99*self.shadow_probe_rho)
            probe=alpha.log()-torch.log1p(-alpha)
            result['opacity']=result['opacity'].index_copy(0,rows[:1],self.support_anchor_raw['opacity'][root:root+1])
            result['opacity']=result['opacity'].index_copy(0,rows[1:],probe)
        return result

    @torch.no_grad()
    def install_details_from_coarse(self, root_indices=None, detail_alpha=0.):
        if bool(self.shadow_accepted.any()):
            raise ValueError('Shadow topology must be installed before accepting a family')
        # Legacy fixed alpha arguments never turn shadows into real density.
        result=super().install_details_from_coarse(root_indices,detail_alpha=0.)
        result.clear_shadow_probe()
        return result

    def capture(self):
        packet=super().capture()
        packet['shadow_source_residual']=packet.pop('quick_source_residual')
        return packet

    def restore(self, packet, training_args, device=None):
        packet=dict(packet);packet['quick_source_residual']=packet['shadow_source_residual']
        return super().restore(packet,training_args,device)
