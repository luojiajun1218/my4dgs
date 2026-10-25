"""ReCon-GS spawn/split and next-frame merge, with our codec row metadata.

Adapted from jyfu-vcl/ReCon-GS, commit
36e87f16fddd5b7e984dba613a3ebb9a3f347f82:
pipeline/ReCon-GS/model/gaussian.py and module.py.
Upstream's opacity initializer writes the *logit* 0.1 (alpha ~0.525).
"""

import torch
from libgs.utils.general import build_rotation

from .birth_quantization import RAW_ATTRIBUTES, RAW_FIELDS
from .gaussian import BirthGaussianModel


def recongs_survivor_rows(opacity, previous_birth_count):
    """Merge last frame's additions, then remove K lowest-alpha rows globally."""
    count = len(opacity)
    if not 0 <= previous_birth_count < count:
        raise ValueError('ReCon-GS merge must retain a nonempty base')
    remove = torch.zeros(count, dtype=torch.bool, device=opacity.device)
    remove[opacity.flatten().argsort(stable=True)[:previous_birth_count]] = True
    return (~remove).nonzero(as_tuple=True)[0]


class ReConGSBirthGaussianModel(BirthGaussianModel):
    @torch.no_grad()
    def initialize_recongs_sources(self, source, step, training_args,
                                   grad_threshold, extent):
        grads = source.xyz_gradient_accum / source.denom
        grads = torch.nan_to_num(grads, nan=0.)
        selected = (torch.linalg.vector_norm(grads, dim=-1) >= grad_threshold)
        selected &= source.get_scaling.amax(dim=-1) <= source.percent_dense * extent
        rows = selected.nonzero(as_tuple=True)[0]
        born = {name: getattr(source, attr)[rows].detach().clone()
                for name, attr in zip(RAW_FIELDS, RAW_ATTRIBUTES)}
        # Upstream samples in world axes, without covariance rotation/shrinking.
        born['xyz'] = torch.normal(born['xyz'], source.get_scaling[rows] * 2.)
        born['opacity'].fill_(.1)
        born['rotation'].zero_()
        born['rotation'][:, 0] = 1.
        self.initialize_from_sources(source, None, step, training_args,
                                     proposal=(born, rows))
        if self.optimizer is not None:
            # Upstream appends to the base, captures/restores that optimizer,
            # then prunes the old rows. New rows have zero moments but retain
            # the base Adam step and group settings. Gather only this metadata
            # instead of cloning the full scene and its moments.
            source_groups = {g['name']: g for g in source.optimizer.param_groups}
            for group in self.optimizer.param_groups:
                original = source_groups[group['name']]
                group.update({k: v for k, v in original.items() if k != 'params'})
                state = source.optimizer.state.get(original['params'][0], {})
                if state:
                    parameter = group['params'][0]
                    self.optimizer.state[parameter] = {
                        k: (torch.zeros_like(parameter) if k in
                            ('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq') else
                            v.clone() if torch.is_tensor(v) else v)
                        for k, v in state.items()}

    @torch.no_grad()
    def split_rows(self, source_rows, created_step, N=2):
        """Upstream split, including native buffer reset even for an empty split.

        Only source_rows/created_steps bookkeeping is added for stream encoding.
        """
        rows = source_rows.to(device=self._xyz.device, dtype=torch.long)
        inherited = self.source_rows[rows].repeat(N)
        stds = self.get_scaling[rows].repeat(N, 1)
        samples = torch.normal(torch.zeros_like(stds), stds)
        rotations = build_rotation(self._rotation[rows]).repeat(N, 1, 1)
        xyz = (torch.bmm(rotations, samples.unsqueeze(-1)).squeeze(-1) +
               self._xyz[rows].repeat(N, 1))
        self.densification_postfix(xyz,
            self._features_dc[rows].repeat(N, 1, 1),
            self._features_rest[rows].repeat(N, 1, 1),
            self._opacity[rows].repeat(N, 1),
            self.scaling_inverse_activation(stds / (.8 * N)),
            self._rotation[rows].repeat(N, 1))
        self.source_rows = torch.cat((self.source_rows, inherited))
        self.created_steps = torch.cat((self.created_steps,
            self.created_steps.new_full((N * len(rows),), int(created_step))))
        remove = torch.zeros(len(self._xyz), dtype=torch.bool, device=self._xyz.device)
        remove[rows] = True
        self.prune_points(remove)
        return N * len(rows)
