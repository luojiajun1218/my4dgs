"""Sparse, quantized base-color updates with an explicit feature stream tag."""
from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
import struct

import torch
from torch import nn

from .legacy_opacity_codec import decode_opacity_delta
from .sparse_updates import OpacityUpdateConfig, QuantizedOpacityUpdate


MAGIC = b'SDC1'
REGIONAL_MAGIC = b'SDR1'


@dataclass
class FeatureDCUpdateConfig:
    enabled: bool = False
    step: float = .01
    learning_rate: float = .0025
    l1_weight: float = .005
    decay_with_motion: bool = True
    joint_with_birth: bool = True
    sharing: str = 'per_gs'
    gate_joint: bool = False

    def scalar_config(self):
        return OpacityUpdateConfig(enabled=self.enabled, step=self.step,
            learning_rate=self.learning_rate, l1_weight=self.l1_weight,
            proximal_regularization=True, decay_with_motion=self.decay_with_motion,
            joint_with_birth=self.enabled and self.joint_with_birth, mode='logit', prune_threshold=0.)

    def validate(self):
        if not isinstance(self.gate_joint, bool):
            raise ValueError('feature_dc_update.gate_joint must be boolean')
        if self.gate_joint and self.sharing != 'spatial_region':
            raise ValueError('Joint gate colors require spatial region sharing')
        if self.sharing not in ('per_gs', 'spatial_region'):
            raise ValueError('Unknown DC feature sharing mode')
        if self.enabled and self.sharing == 'spatial_region':
            raise ValueError('Regional shared-colour training is retired; historical packets remain decodable')
        if not isinstance(self.joint_with_birth, bool):
            raise ValueError('feature_dc_update.joint_with_birth must be boolean')
        self.scalar_config().validate()


def _regional_owner(owner, node_count, rows):
    if (isinstance(node_count, bool) or not isinstance(node_count, int) or node_count <= 0 or
            owner is None):
        raise ValueError('Regional DC updates require the decoded motion owner and node count')
    owner = torch.as_tensor(owner).detach().cpu().contiguous()
    if (owner.dtype != torch.long or owner.shape != (rows,) or
            bool(((owner < 0) | (owner >= node_count)).any())):
        raise ValueError('Regional DC ownership differs from the surviving GS layout')
    return owner


def _regional_digest(reference, owner, node_count):
    digest = hashlib.sha256(struct.pack('<QQ', node_count, len(reference)))
    digest.update(reference.detach().cpu().contiguous().numpy().tobytes())
    digest.update(owner.detach().cpu().contiguous().numpy().tobytes())
    return digest.digest()


def _merge_regional_dc(reference, delta):
    # A zero update must preserve signed zero and all other reference bits.
    return torch.where(delta.ne(0), reference + delta, reference)


def decode_feature_dc_delta(packet, reference, *, frame_index, owner=None, node_count=None):
    reference = torch.as_tensor(reference).detach().cpu().contiguous()
    if reference.ndim != 3 or reference.shape[1:] != (1, 3):
        raise ValueError('DC features must have shape [N, 1, 3]')
    if packet[:4] == REGIONAL_MAGIC:
        owner = _regional_owner(owner, node_count, len(reference))
        if len(packet) < 36 or packet[4:36] != _regional_digest(reference, owner, node_count):
            raise ValueError('Regional DC reference or motion ownership differs')
        zeros = reference.new_zeros(node_count * 3, 1)
        decoded = decode_opacity_delta(packet[36:], zeros, frame_index=frame_index)
        if decoded.mode != 'logit':
            raise ValueError('Regional DC updates require additive linear symbols')
        delta = decoded.logits.reshape(node_count, 1, 3)[owner]
        return _merge_regional_dc(reference, delta)
    if packet[:4] != MAGIC:
        raise ValueError('Unknown DC feature update stream')
    decoded = decode_opacity_delta(packet[4:], reference.reshape(-1, 1), frame_index=frame_index)
    if decoded.mode != 'logit':
        raise ValueError('DC updates require additive linear symbols')
    return decoded.logits.reshape_as(reference)


class QuantizedFeatureDCUpdate(nn.Module):
    """Per-GS DC increments. SDR1 regional packets are supported for decode only."""
    def __init__(self, config, motion=None):
        super().__init__()
        config.validate()
        if config.sharing != 'per_gs' or config.gate_joint:
            raise ValueError('Regional shared-colour training is retired')
        self.config = config
        self.scalar = QuantizedOpacityUpdate(config.scalar_config())

    @property
    def regional(self):
        return False

    def setup(self, reference):
        if reference.ndim != 3 or reference.shape[1:] != (1, 3):
            raise ValueError('DC features must have shape [N, 1, 3]')
        self.scalar.setup(reference.reshape(-1, 1))

    def forward(self, training=False):
        return self.scalar().reshape(-1, 1, 3)

    def canonical_features(self):
        return self.scalar.canonical_logits().reshape(-1, 1, 3)

    def rate_terms(self, byte_price, width=1.):
        terms = self.scalar.rate_terms(byte_price, width)
        tag = terms['active_rows'].gt(0).to(terms['loss']) * len(MAGIC)
        return {**terms, 'loss': terms['loss'] + tag * byte_price,
                'raw_bytes_surrogate': terms['raw_bytes_surrogate'] + tag}

    def optimizer_step(self, *args, **kwargs):
        self.scalar.optimizer_step(*args, **kwargs)

    def commit(self, rows):
        channels = torch.arange(3, device=rows.device)
        self.scalar.commit((rows[:, None] * 3 + channels).flatten())

    def finish(self):
        self.scalar.finish()

    def capture(self):
        if self.scalar.reference is None:
            return None
        return dict(config=asdict(self.config), scalar=self.scalar.capture())

    def restore(self, packet, device):
        normalized = None if packet is None else {**asdict(FeatureDCUpdateConfig()), **packet['config']}
        if packet is None or normalized != asdict(self.config):
            raise ValueError('DC feature checkpoint configuration differs or is missing')
        self.scalar.restore(packet['scalar'], device)
        if len(self.scalar.rows) % 3:
            raise ValueError('Incomplete RGB channel rows')
        rows = self.scalar.rows.reshape(-1, 3)
        if not torch.equal(rows, (rows[:, :1] // 3) * 3 + torch.arange(3, device=device)):
            raise ValueError('DC feature rows lost GS/channel alignment')

    @torch.no_grad()
    def export(self, path, frame_index):
        path = Path(path)
        info = self.scalar.export(path, frame_index)
        if info is None:
            return None
        packet = MAGIC + path.read_bytes()
        reference = self.scalar.reference[self.scalar.rows].reshape(-1, 1, 3)
        decoded = decode_feature_dc_delta(packet, reference, frame_index=frame_index)
        if not torch.equal(decoded, self.canonical_features()):
            raise ValueError('DC feature packet differs from materialized features')
        path.write_bytes(packet)
        return {**info, 'archive_bytes': len(packet), 'format': packet[:4].decode(),
                'channels': 3, 'sharing': 'per_gs', 'timing': 'before_births'}
