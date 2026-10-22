"""Shared tiny Gaussian rows and production stream export for decoder regressions."""
import torch

from pipeline.module import BirthConfig
from pipeline.model.birth_quantization import (RAW_FIELDS, RAW_ATTRIBUTES,
    QuantizedNewGaussianView, materialize_quantized_gaussians)
from pipeline.model.rigid_residual_tree import RigidTreeConfig
from pipeline.model.spatial_gated_motion import SpatialGatedMotion
from pipeline.model.rigid_tree_birth_codec import export_birth_attributes


class Rows:
    def __init__(self, raw):
        self.prune_calls = 0
        for name, attr in zip(RAW_FIELDS, RAW_ATTRIBUTES):
            setattr(self, attr, raw[name].detach().clone())

    def prune_points(self, remove):
        self.prune_calls += 1
        for attr in RAW_ATTRIBUTES:
            setattr(self, attr, getattr(self, attr)[~remove].clone())


def reference_raw():
    xyz = torch.cartesian_prod(*[torch.tensor([-1., 1.])] * 3)
    ids = torch.arange(len(xyz), dtype=torch.float32)
    return dict(xyz=xyz, rotation=torch.tensor([[1., 0., 0., 0.]] * len(xyz)),
        logscale=ids[:, None].expand(-1, 3).clone() * .03,
        opacity=torch.logit(torch.tensor([[.005]] + [[.7]] * (len(xyz) - 1))),
        sh_dc=ids[:, None, None].expand(-1, 1, 3).clone() * .1,
        sh_rest=ids[:, None, None].expand(-1, 3, 3).clone() * .01)


def motion_model(raw):
    cfg = RigidTreeConfig()
    cfg.spatial_gate.max_depth = 2
    cfg.spatial_gate.min_region_size = 2
    cfg.spatial_gate.training_schedule = 'joint'
    cfg.spatial_gate.start_step = cfg.spatial_gate.gate_start_step = 0
    cfg.spatial_gate.initial_logit = -2.
    cfg.validate()
    motion = SpatialGatedMotion(cfg)
    motion.setup(raw['xyz'], raw['rotation'], 2., frame_index=1)
    return motion


def export_frame(directory, motion, raw, sources):
    xyz, rotation = motion.materialized_pose()
    base = {**raw, 'xyz': xyz, 'rotation': rotation}
    newborn = {name: value[sources].clone() for name, value in base.items()}
    if len(sources):
        newborn['xyz'][:, 0] += .003
        newborn['opacity'][:] = torch.logit(torch.tensor(.8))
        if len(sources) > 1:
            newborn['opacity'][-1] = torch.logit(torch.tensor(.001))
        newborn['sh_dc'] += .01
    view = (QuantizedNewGaussianView(Rows(newborn), BirthConfig(), 2., base, sources)
            if len(sources) else None)
    expected, _ = materialize_quantized_gaussians(base, view)
    motion.register_materialized_layout(expected['xyz'], expected['rotation'], sources)
    frame = motion.frame_index
    motion.export_motion(directory / f'motion_{frame:03d}.bin')
    expected['xyz'] = motion.codec_previous_state.xyz
    expected['rotation'] = motion.codec_previous_state.rotation
    if len(sources):
        export_birth_attributes(directory / f'birth_{frame:03d}.bin',
            {name: expected[name][-len(sources):] for name in RAW_FIELDS[2:]},
            len(raw['xyz']),
            symbols={name: view.symbols[name] for name in RAW_FIELDS[2:]},
            steps={name: view.steps[name] for name in RAW_FIELDS[2:]},
            predictor={name: view.predictor[name] for name in RAW_FIELDS[2:]},
            source_rows=sources)
    return expected
