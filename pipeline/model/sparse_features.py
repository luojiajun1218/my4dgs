"""Sparse, quantized base-color updates with an explicit feature stream tag."""
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn

from .legacy_opacity_codec import decode_opacity_delta
from .sparse_updates import OpacityUpdateConfig, QuantizedOpacityUpdate


MAGIC = b'SDC1'


@dataclass
class FeatureDCUpdateConfig:
    enabled: bool = False
    step: float = .01
    learning_rate: float = .0025
    l1_weight: float = .005
    decay_with_motion: bool = True
    joint_with_birth: bool = True

    def scalar_config(self):
        return OpacityUpdateConfig(enabled=self.enabled, step=self.step,
            learning_rate=self.learning_rate, l1_weight=self.l1_weight,
            proximal_regularization=True, decay_with_motion=self.decay_with_motion,
            joint_with_birth=self.enabled and self.joint_with_birth, mode='logit', prune_threshold=0.)

    def validate(self):
        if not isinstance(self.joint_with_birth, bool):
            raise ValueError('feature_dc_update.joint_with_birth must be boolean')
        self.scalar_config().validate()


def decode_feature_dc_delta(packet, reference, *, frame_index):
    reference = torch.as_tensor(reference).detach().cpu().contiguous()
    if reference.ndim != 3 or reference.shape[1:] != (1, 3):
        raise ValueError('DC features must have shape [N, 1, 3]')
    if packet[:4] != MAGIC:
        raise ValueError('Unknown DC feature update stream')
    decoded = decode_opacity_delta(packet[4:], reference.reshape(-1, 1), frame_index=frame_index)
    if decoded.mode != 'logit':
        raise ValueError('DC updates require additive linear symbols')
    return decoded.logits.reshape_as(reference)


class QuantizedFeatureDCUpdate(nn.Module):
    """Reuse the scalar additive coder; rows are GS-major RGB channels."""
    def __init__(self, config):
        super().__init__()
        config.validate()
        self.config = config
        self.scalar = QuantizedOpacityUpdate(config.scalar_config())

    def setup(self, reference):
        if reference.ndim != 3 or reference.shape[1:] != (1, 3):
            raise ValueError('DC features must have shape [N, 1, 3]')
        self.scalar.setup(reference.reshape(-1, 1))

    def forward(self):
        return self.scalar().reshape(-1, 1, 3)

    def canonical_features(self):
        return self.scalar.canonical_logits().reshape(-1, 1, 3)

    def rate_terms(self, byte_price, width=1.):
        terms = self.scalar.rate_terms(byte_price, width)
        # One explicit field tag is also present in the actual archive.
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
        if packet is None or packet['config'] != asdict(self.config):
            raise ValueError('DC feature checkpoint configuration differs or is missing')
        self.scalar.restore(packet['scalar'], device)
        if len(self.scalar.rows) % 3:
            raise ValueError('Incomplete RGB channel rows')
        rows = self.scalar.rows.reshape(-1, 3)
        if not torch.equal(rows, (rows[:, :1] // 3) * 3 + torch.arange(3, device=device)):
            raise ValueError('DC feature rows lost GS/channel alignment')

    @torch.no_grad()
    def export(self, path, frame_index):
        # The scalar export validates exact values against its decoder first.
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
        return {**info, 'archive_bytes': len(packet), 'format': MAGIC.decode(),
                'channels': 3, 'timing': 'before_births'}
