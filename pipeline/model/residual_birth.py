"""Shared adaptive-birth state, spatial grouping, and CPU codec trials."""

from dataclasses import dataclass, field
import math

import torch
from torch.nn import functional as F

@dataclass
class ResidualBirthConfig:
    proposal_views: int = 8
    validation_views: int = 4  # Legacy config compatibility; v5 uses every training view.
    max_candidates: int = 512
    repair_candidates: int = 128
    depth_planes: int = 64
    max_groups: int = 16
    birth_budget_bytes: int = 16000
    target_frame_bytes: int = 50000
    max_gaussians: int = 150000
    residual_floor: float = .03
    match_cost_limit: float = .25
    voxel_extent: float = .002
    validation_tolerance: float = 1.e-4
    max_cohorts: int = 2

    def validate(self):
        for name in ('proposal_views', 'validation_views', 'max_candidates',
                     'depth_planes', 'max_groups', 'birth_budget_bytes',
                     'target_frame_bytes', 'max_gaussians', 'max_cohorts'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'birth.residual.{name} must be a positive integer')
        if not 0 <= self.repair_candidates < self.max_candidates:
            raise ValueError('Repair quota must leave room for new spatial support')
        if self.depth_planes < 8:
            raise ValueError('RGB depth search needs at least eight planes')
        for name in ('residual_floor', 'match_cost_limit', 'voxel_extent'):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'birth.residual.{name} must be positive')
        if not 0 <= self.validation_tolerance < .01:
            raise ValueError('Invalid independent-camera tolerance')


def spatial_groups(xyz, limit):
    """Balanced spatial groups; validation evaluates whole interacting groups."""
    if not len(xyz):
        return []
    groups = [torch.arange(len(xyz), device=xyz.device)]
    while len(groups) < limit:
        sizes = [len(group) for group in groups]
        index = max(range(len(groups)), key=sizes.__getitem__)
        ids = groups[index]
        if len(ids) < 2:
            break
        axis = xyz[ids].amax(0).sub(xyz[ids].amin(0)).argmax()
        ordered = ids[xyz[ids, axis].argsort(stable=True)]
        middle = len(ordered)//2
        groups[index:index+1] = [ordered[:middle], ordered[middle:]]
    return groups


class SubsetView:
    def __init__(self, model, rows):
        self.model, self.rows = model, rows

    def __getattr__(self, name):
        return getattr(self.model, name)

    @property
    def get_xyz(self): return self.model.get_xyz[self.rows]
    @property
    def get_rotation(self): return self.model.get_rotation[self.rows]
    @property
    def get_scaling(self): return self.model.get_scaling[self.rows]
    @property
    def get_opacity(self): return self.model.get_opacity[self.rows]
    @property
    def get_features(self): return self.model.get_features[self.rows]

    def get_covariance(self, scaling_modifier=1.):
        return self.model.get_covariance(scaling_modifier)[self.rows]


@dataclass
class BirthTransaction:
    raw: dict
    sources: torch.Tensor
    replacements: torch.Tensor
    scores: torch.Tensor
    seed_xyz: torch.Tensor
    seed_rotation: torch.Tensor
    seed_scale: torch.Tensor
    observations: torch.Tensor
    fit_names: tuple
    validation_names: tuple
    proposed_spatial: int = 0
    proposed_repairs: int = 0
    group_ids: torch.Tensor = None
    region_diagnostics: dict = field(default_factory=dict)
    support_domains: list = field(default_factory=list)
    provisional_deleted: torch.Tensor = None
    fixed_keep: torch.Tensor = None
    fixed_reference: dict = None
    family_state: dict = None
    patch_state: dict = None
    patch_specs: list = field(default_factory=list)
    old_base_symbols: torch.Tensor = None
    old_base_opacity: torch.Tensor = None
    fixed_old_symbols: torch.Tensor = None
    old_render_reference: dict = None
    reincarnation_state: dict = None
    reincarnation_specs: list = field(default_factory=list)
    crossfit_state: dict = None
    predictive_model: bool = False
    persistent_state: dict = None

    @staticmethod
    def tensor_tree(value, device):
        if torch.is_tensor(value):
            return value.detach().to(device).clone()
        if isinstance(value, dict):
            return {key: BirthTransaction.tensor_tree(item, device) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return type(value)(BirthTransaction.tensor_tree(item, device) for item in value)
        return value

    def capture(self):
        return self.tensor_tree(self.__dict__, 'cpu')

    @classmethod
    def restore(cls, state, device):
        return cls(**cls.tensor_tree(state, device))


class RawView:
    def __init__(self, raw, degree, basis, alpha=None):
        self.get_xyz = raw['xyz']
        self.get_rotation = F.normalize(raw['rotation'], dim=-1)
        self.get_scaling = raw['logscale'].exp()
        self.get_opacity = raw['opacity'].sigmoid() if alpha is None else alpha
        self.get_features = torch.cat((raw['sh_dc'], raw['sh_rest']), 1)
        self.active_sh_degree = self.max_sh_degree = degree
        self.basis = basis

    def get_covariance(self, scaling_modifier=1.):
        return self.basis.covariance_activation(self.get_scaling, scaling_modifier,
                                                self.get_rotation)


@dataclass
class CommitTrial:
    keep: torch.Tensor
    candidates: torch.Tensor
    sources: torch.Tensor
    raw: dict
    complete_bytes: int
    motion_bytes: int
    birth_bytes: int
    opacity_bytes: int
    old_symbols: torch.Tensor = None

    def capture(self):
        return BirthTransaction.tensor_tree(self.__dict__, 'cpu')

    @classmethod
    def restore(cls, state, device):
        return cls(**BirthTransaction.tensor_tree(state, device))


@torch.no_grad()
def encode_trial(module, candidate_mask, delete_mask, encode=True):
    """Dry-run the real codecs without advancing any training/decoder state."""
    from .birth_quantization import model_raw, QuantizedNewGaussianView, materialize_quantized_gaussians
    from .rigid_tree_codec import encode_motion, decode_motion
    from .rigid_tree_birth_codec import encode_birth_symbols, encode_birth_attributes, ATTRIBUTES
    from .sparse_updates import encode_opacity_delta
    selected = candidate_mask.nonzero(as_tuple=True)[0]
    transaction = module.birth_transaction
    fixed = transaction.fixed_reference
    candidate = module.births.subset(selected)
    delete = delete_mask.clone()
    keep = (~delete).nonzero(as_tuple=True)[0]
    if not len(keep):
        raise ValueError('A transaction must retain an incumbent prediction reference')
    if fixed is not None:
        if not torch.equal(keep, transaction.fixed_keep):
            raise ValueError('Population refit cannot change its selected old-row structure')
        base = {name: value.detach().clone() for name, value in fixed.items()}
        sources = transaction.sources[selected].clone()
    else:
        reference = (transaction.old_render_reference if transaction.old_render_reference is not None
                     else model_raw(module.gaussians))
        base = {name: value[keep] for name, value in reference.items()}
        mapping = torch.full((len(delete),), -1, device=keep.device, dtype=torch.long)
        mapping[keep] = torch.arange(len(keep), device=keep.device)
        sources = mapping[transaction.sources[selected]]
        if bool((sources < 0).any()):
            raise ValueError('Independent birth codec parent was deleted')
    old_symbols = None
    view = (QuantizedNewGaussianView(candidate, module.config.birth,
            module.datamodule.cameras_extent, base, sources) if len(selected) else None)
    raw, _ = materialize_quantized_gaussians(base, view)
    if not encode:
        return CommitTrial(keep, selected, sources, raw, 0, 0, 0, 0, old_symbols)
    motion = module.motion
    rt, rr, gt, gr = [value.detach().cpu() for value in motion._values(hard=True)]
    packet = encode_motion(motion.frame_reference_state, motion.frame_index, motion.extent,
        motion.previous_rows[keep].cpu(), motion.owner[keep].cpu(), motion.parents.cpu(),
        rt, rr, gt[keep.cpu()], gr[keep.cpu()], sources.cpu(),
        raw['xyz'][len(keep):].cpu(), raw['rotation'][len(keep):].cpu(),
        motion.config.motion_xyz_step_extent, motion.config.motion_rotation_step_rad,
        compression='auto', gs_translation_step_extent=motion.config.gs_xyz_step_extent,
        gs_rotation_step_rad=motion.config.gs_rotation_step_rad,
        spatial_open=motion.opened.cpu(), spatial_relative=motion.relative_regions,
        inherit_birth_rotation=False,
        birth_pose_steps=(module.config.birth.new_xyz_step_extent*module.datamodule.cameras_extent,
                          module.config.birth.new_rotation_step))
    decoded = decode_motion(packet, motion.frame_reference_state)
    raw['xyz'] = decoded.xyz.to(module.device)
    raw['rotation'] = decoded.rotation.to(module.device)
    birth_packet = b''
    if view is not None:
        appearance = {name: raw[name][len(keep):] for name in ATTRIBUTES}
        birth_packet = encode_birth_symbols(appearance, len(keep),
            symbols={name: view.symbols[name] for name in ATTRIBUTES},
            steps={name: view.steps[name] for name in ATTRIBUTES},
            predictor={name: view.predictor[name] for name in ATTRIBUTES},
            source_rows=sources)
        if birth_packet is None:
            birth_packet = encode_birth_attributes(appearance, len(keep))
    opacity_packet = b''
    if module.opacity_update is not None:
        update = module.opacity_update
        rows = update.rows[keep]
        opacity_packet = encode_opacity_delta(update.reference[rows],
            update.symbols()[rows], frame_index=module.current_frame,
            step=update.step, mode=update.config.mode) or b''
    # Reserve more than the current manifest actually uses; actual final files
    # remain the authoritative storage measurement.
    size = len(packet)+len(birth_packet)+len(opacity_packet)+1024
    return CommitTrial(keep, selected, sources, raw, size,
                        len(packet), len(birth_packet), len(opacity_packet), old_symbols)
