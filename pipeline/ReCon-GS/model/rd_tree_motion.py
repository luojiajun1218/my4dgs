"""Trainable tree-envelope motion, separate from existing deformation modes."""
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from numbers import Integral
from typing import Optional

import numpy as np
import torch
from torch import nn

from .motion_codec import rotation_vector_to_quaternion
from .rd_tree_projection import (build_candidate_tree, project_rd_tree, encode_projection,
                                 decode_projection, ProjectionWorkspace)


@dataclass
class RDTreeConfig:
    max_depth: int = 13
    fixed_cut_depth: Optional[int] = None
    structure_warmup_updates: int = 0
    min_leaf_size: int = 24
    partition: str = 'geometry_2means'
    clustering_iterations: int = 4
    quant_xyz: float = .0001
    quant_rotation: float = .001
    rho_xyz: float = .001
    rho_rotation: float = .01
    normalized_rate: float = .0001
    step_fraction: float = .5
    accumulation_steps: int = 4
    rebuild_tree_on_export: bool = True


class _AccumulatedStep:
    """Adapter for the pipeline's per-view step()/zero_grad() convention."""
    def __init__(self, owner):
        self.owner = owner

    def step(self):
        self.owner.accumulate_and_step()

    def zero_grad(self, set_to_none=True):
        if set_to_none:
            self.owner.latent.grad = None
        elif self.owner.latent.grad is not None:
            self.owner.latent.grad.zero_()


class RDTreeMotion(nn.Module):
    def __init__(self, config: RDTreeConfig):
        super().__init__()
        self.config = config
        self.settings = config.rd_tree if hasattr(config, 'rd_tree') else config
        config = self.settings
        if config.fixed_cut_depth is not None and (isinstance(config.fixed_cut_depth, bool)
                or not isinstance(config.fixed_cut_depth, Integral) or config.fixed_cut_depth < 0):
            raise ValueError('Invalid fixed hierarchy depth')
        if (isinstance(config.structure_warmup_updates, bool)
                or not isinstance(config.structure_warmup_updates, Integral)
                or config.structure_warmup_updates < 0
                or (config.structure_warmup_updates and config.fixed_cut_depth is not None)):
            raise ValueError('Invalid or conflicting structure warmup')
        positive = (config.quant_xyz, config.quant_rotation, config.rho_xyz,
                    config.rho_rotation, config.step_fraction)
        if (not all(np.isfinite(x) and x > 0 for x in positive)
                or not np.isfinite(config.normalized_rate) or config.normalized_rate < 0
                or config.accumulation_steps < 1):
            raise ValueError('Invalid tree motion optimization parameters')
        if hasattr(self.config, 'num_stage1_steps') and self.config.num_stage1_steps % config.accumulation_steps:
            raise ValueError('Motion stage must contain complete accumulated gradient batches')
        if (hasattr(self.config, 'num_stage1_steps') and config.structure_warmup_updates
                > self.config.num_stage1_steps // config.accumulation_steps):
            raise ValueError('Structure warmup must end by motion-stage finalization')
        self.initialized = False
        self.optimizer = _AccumulatedStep(self)

    def _buffer(self, name, value):
        if name in self._buffers:
            setattr(self, name, value)
        else:
            self.register_buffer(name, value)

    def setup(self, xyz, reset_grid=False):
        # Encoder-only latent/moments are not motion-stream storage.
        self.tree = build_candidate_tree(xyz.detach().cpu().double().numpy(),
                                         self.settings.max_depth, self.settings.min_leaf_size,
                                         self.settings.partition, self.settings.clustering_iterations)
        self.workspace = ProjectionWorkspace(self.tree)
        self._buffer('xyz', xyz.detach().clone())
        self.latent = nn.Parameter(torch.zeros((len(xyz), 6), dtype=torch.float64, device=xyz.device))
        self._buffer('pending_gradient', torch.zeros_like(self.latent))
        self._buffer('decoded', torch.zeros_like(self.latent))
        self._buffer('rho', self.latent.new_tensor([self.settings.rho_xyz] * 3
                                                  + [self.settings.rho_rotation] * 3))
        self.quant_step = np.array([self.settings.quant_xyz] * 3 + [self.settings.quant_rotation] * 3)
        self.rate = self.settings.normalized_rate / len(xyz)
        self.pending_count = self.updates = self.projection_calls = 0
        self.projection = None
        self.dirty = True
        self.initialized = True

    def reset(self, xyz):
        self.setup(xyz)

    def reset_grid(self, xyz):
        self.setup(xyz)

    def collecting_gradients(self, step):
        return False  # Dense latent gradients already expose inactive details.

    @property
    def get_grid_num(self):
        return 1 + len(self.project().split_nodes) if self.initialized else 0

    def reg_loss(self):
        return self.field().abs().mean()

    @torch.no_grad()
    def project(self):
        if not self.initialized:
            raise RuntimeError('Motion must be initialized before projection')
        if self.dirty:
            self.projection = project_rd_tree(self.latent.detach().cpu().numpy(), self.tree,
                                              self.quant_step, self.rate, self.rho.cpu().numpy(),
                                              vectorized_symbols=True, vectorized_statistics=True,
                                              workspace=self.workspace,
                                              fixed_cut_depth=self.effective_fixed_cut_depth)
            self.decoded.copy_(torch.as_tensor(self.projection.field, device=self.latent.device))
            self.projection_calls += 1
            self.dirty = False
        return self.projection

    @property
    def effective_fixed_cut_depth(self):
        if self.updates < self.settings.structure_warmup_updates:
            return self.settings.max_depth
        return self.settings.fixed_cut_depth

    def field(self):
        self.project()
        # Subtract identical latent values first: avoid round-off cancellation
        # in x + stop_gradient(z-x), so the forward value is exactly decoded z.
        return self.decoded + (self.latent - self.latent.detach())

    def forward(self, xyz):
        if len(xyz) != len(self.xyz):
            raise ValueError('Gaussian count changed without rebuilding the candidate tree')
        value = self.field().to(dtype=xyz.dtype)
        return value[:, :3].contiguous(), rotation_vector_to_quaternion(value[:, 3:])

    def envelope_loss(self):
        projection = self.project()
        return .5 * (self.rho * (self.latent - self.decoded).square()).sum() + self.rate * projection.payload_bits

    @torch.no_grad()
    def accumulate_and_step(self):
        if self.latent.grad is None:
            raise RuntimeError('Expected a motion gradient for this training view')
        if not torch.isfinite(self.latent.grad).all():
            raise FloatingPointError('Nonfinite motion gradient')
        self.pending_gradient.add_(self.latent.grad)
        self.pending_count += 1
        if self.pending_count == self.settings.accumulation_steps:
            self.latent.add_(-self.settings.step_fraction / self.rho
                             * (self.pending_gradient / self.pending_count))
            self.pending_gradient.zero_()
            self.pending_count = 0
            self.updates += 1
            self.dirty = True

    def capture(self):
        if not self.initialized:
            return {}
        projection = self.project()
        return {'rd_tree_version': 1, 'config': asdict(self.settings),
                'state': {name: value.detach().clone() for name, value in self.state_dict().items()},
                'pending_count': self.pending_count, 'updates': self.updates,
                'projection_calls': self.projection_calls,
                'packet': encode_projection(projection, self.tree)}

    def restore(self, checkpoint):
        if not checkpoint:
            return
        saved_config = asdict(RDTreeConfig(**checkpoint['config']))
        if checkpoint.get('rd_tree_version') != 1 or saved_config != asdict(self.settings):
            raise ValueError('Incompatible tree motion checkpoint/configuration')
        self.setup(checkpoint['state']['xyz'])
        self.load_state_dict(checkpoint['state'])
        self.pending_count = int(checkpoint['pending_count'])
        if not 0 <= self.pending_count < self.settings.accumulation_steps:
            raise ValueError('Invalid accumulated gradient count')
        self.updates = int(checkpoint['updates'])
        self.dirty = True
        if encode_projection(self.project(), self.tree) != checkpoint['packet']:
            raise ValueError('Restored projection does not match checkpoint packet')
        self.projection_calls = int(checkpoint['projection_calls'])

    @torch.no_grad()
    def export(self, path):
        if self.pending_count:
            raise RuntimeError('Cannot finalize a frame with a partial gradient batch')
        if self.updates < self.settings.structure_warmup_updates:
            raise RuntimeError('Cannot export before adaptive structure selection activates')
        path = Path(path)
        projection = self.project()  # Include the last optimizer update.
        packet = encode_projection(projection, self.tree)
        protocol = {'rd_tree_protocol': 1, 'max_depth': self.settings.max_depth,
                    'min_leaf_size': self.settings.min_leaf_size, 'partition': self.settings.partition,
                    'clustering_iterations': self.settings.clustering_iterations,
                    'geometry': 'previous decoded base after deterministic merge/prune; original point order',
                    'arithmetic': 'CPU float64 candidate tree; ties follow build_candidate_tree'}
        # Write only new targets; never silently overwrite an existing stream.
        metadata_path = path.with_suffix(path.suffix + '.json')
        if path.exists() or metadata_path.exists():
            raise FileExistsError('Motion export target already exists')
        path.write_bytes(packet)
        metadata_path.write_text(json.dumps(protocol, sort_keys=True, indent=2) + '\n')
        decoded = (decode_tree_motion(packet, self.xyz.detach().cpu().double().numpy(), protocol)
                   if self.settings.rebuild_tree_on_export else decode_projection(packet, self.tree))
        if not np.array_equal(decoded, projection.field):
            raise AssertionError('Packet decode differs from encoded motion')
        return {'payload_bits': projection.payload_bits, 'splits': len(projection.split_nodes),
                'packet_bytes': len(packet), 'protocol_bytes': metadata_path.stat().st_size,
                'motion_bytes': len(packet) + metadata_path.stat().st_size,
                'updates': self.updates, 'projection_calls': self.projection_calls,
                'export_verification': ('independent_tree_rebuild' if self.settings.rebuild_tree_on_export
                                        else 'existing_candidate_tree')}


def decode_tree_motion(packet, previous_xyz, protocol):
    if protocol.get('rd_tree_protocol') != 1:
        raise ValueError('Unknown candidate tree protocol')
    tree = build_candidate_tree(previous_xyz, protocol['max_depth'], protocol['min_leaf_size'],
                                protocol['partition'], protocol['clustering_iterations'])
    return decode_projection(packet, tree)
