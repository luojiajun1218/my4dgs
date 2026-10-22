"""Small motion fixtures shared by CPU regression tests; no TestCase dependencies."""
import copy
from functools import lru_cache
from pathlib import Path
import tempfile

import torch
import yaml

from pipeline.model.rigid_residual_tree import RigidTreeConfig
from pipeline.model.spatial_gate import SpatialGateConfig
from pipeline.model.spatial_gated_motion import SpatialGatedMotion
from pipeline.model.rigid_tree_codec import encode_motion, decode_motion


def joint_motion(initial_logit=-2.):
    axis = torch.linspace(-1., 1., 8)
    xyz = torch.cat((torch.cartesian_prod(axis, axis, axis),
                     torch.tensor([[-100., -100., -100.], [100., 100., 100.]])))
    q = torch.zeros(len(xyz), 4)
    q[:, 0] = 1.
    config = RigidTreeConfig(motion_xyz_step_extent=.001, gs_xyz_step_extent=.001,
        motion_rotation_step_rad=.001, gs_rotation_step_rad=.001,
        rate_loss_weight=.002, relative_motion_loss_weight=.01,
        rate_warmup_steps=0, rate_ramp_steps=1, motion_fit_window_steps=20)
    gate = config.spatial_gate
    gate.max_depth = 3
    gate.initialization = 'robust_median'
    gate.initialization_distance_quantile = .99
    gate.outlier_assignment = 'nearest_region'
    gate.training_schedule = 'joint'
    gate.learn_region_motion = True
    gate.min_region_size = 2
    gate.initial_logit = initial_logit
    gate.start_step = gate.gate_start_step = 0
    gate.freeze_step = 70
    config.validate()
    model = SpatialGatedMotion(config)
    model.setup(xyz, q, 2., frame_index=1)
    return model


def staged_motion(repetitions=16, extent=1.):
    coords = torch.cartesian_prod(torch.tensor([.1, .3, .6, .9]),
                                 torch.tensor([.1, .3, .6, .9]),
                                 torch.tensor([.1, .3, .6, .9]))
    xyz = coords.repeat_interleave(repetitions, 0)
    q = torch.zeros(len(xyz), 4); q[:, 0] = 1
    cfg = RigidTreeConfig(region_count=9,
        motion_xyz_step_extent=.001, gs_xyz_step_extent=.001,
        motion_rotation_step_rad=.001, gs_rotation_step_rad=.001,
        rate_loss_weight=.015, relative_motion_loss_weight=0.)
    cfg.spatial_gate.enabled = True
    cfg.spatial_gate.max_depth = 3
    cfg.spatial_gate.min_region_size = 8
    model = SpatialGatedMotion(cfg)
    model.setup(xyz, q, extent, frame_index=1)
    return model


def commit_motion(model, births=None, sources=None):
    reference = copy.deepcopy(model.frame_reference_state)
    xyz, q = model.materialized_pose()
    sources = torch.empty(0, dtype=torch.long) if sources is None else sources
    if births is not None:
        xyz = torch.cat((xyz, births))
        q = torch.cat((q, q[sources]))
    model.register_materialized_layout(xyz, q, sources)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'motion.bin'
        model.export_motion(path)
        decoded = decode_motion(path.read_bytes(), reference)
    torch.testing.assert_close(decoded.xyz, xyz, rtol=0, atol=1e-6)
    torch.testing.assert_close(decoded.rotation, q, rtol=0, atol=1e-6)
    return decoded


@lru_cache(maxsize=1)
def _current_recipe():
    path = Path(__file__).resolve().parents[1] / 'config/n3dv/flame_steak_1020_full300.yaml'
    return yaml.safe_load(path.read_text())['module']['rigid_tree']


def current_motion(points_per_axis=4, *, max_depth=None, amplitude_refit_steps=None):
    """Use the formal recipe on a tiny synthetic point cloud, without CUDA or images."""
    values = copy.deepcopy(_current_recipe())
    gate = SpatialGateConfig(**values.pop('spatial_gate'))
    if max_depth is not None:
        gate.max_depth = max_depth
    if amplitude_refit_steps is not None:
        values['amplitude_refit_steps'] = amplitude_refit_steps
    config = RigidTreeConfig(**values, spatial_gate=gate)
    config.validate()
    axis = torch.linspace(-1., 1., points_per_axis)
    xyz = torch.cartesian_prod(axis, axis, axis)
    rotation = torch.zeros(len(xyz), 4)
    rotation[:, 0] = 1.
    motion = SpatialGatedMotion(config)
    motion.setup(xyz, rotation, 2., frame_index=1)
    motion.observations.fill_(1)
    return motion


def motion_packet(model, compression='auto'):
    rt, rr, gt, gr = model._values(hard=True)
    config = model.config
    return encode_motion(model.frame_reference_state, model.frame_index, model.extent,
        model.previous_rows, model.owner, model.parents, rt, rr, gt, gr,
        torch.empty(0, dtype=torch.long), torch.empty(0, 3), torch.empty(0, 4),
        config.motion_xyz_step_extent, config.motion_rotation_step_rad,
        compression=compression, gs_translation_step_extent=config.gs_xyz_step_extent,
        gs_rotation_step_rad=config.gs_rotation_step_rad, spatial_open=model.opened,
        spatial_relative=model.relative_regions, inherit_birth_rotation=not config.rotation_updates)
