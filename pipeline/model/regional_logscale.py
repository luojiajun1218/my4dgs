"""Optional quantized regional log-scale increments on the frozen motion cut."""
from dataclasses import dataclass
from pathlib import Path
import torch
from .sparse_features import (FeatureDCUpdateConfig, RegionalFeatureDCUpdate,
    REGIONAL_MAGIC, decode_feature_dc_delta, _regional_digest)
from .sparse_updates import encode_opacity_delta

MAGIC = b'RLS1'

@dataclass
class RegionalLogScaleConfig:
    enabled: bool = False
    step: float = .005
    learning_rate: float = .005
    l1_weight: float = .005
    def validate(self):
        if not isinstance(self.enabled, bool):
            raise ValueError('regional_logscale.enabled must be boolean')
        self.dc_config().validate()
    def dc_config(self):
        return FeatureDCUpdateConfig(enabled=self.enabled, step=self.step,
            learning_rate=self.learning_rate, l1_weight=self.l1_weight,
            sharing='spatial_region', post_motion_only=self.enabled)

class RegionalLogScaleUpdate(RegionalFeatureDCUpdate):
    def __init__(self, config, motion):
        config.validate()
        super().__init__(config.dc_config(), motion=motion)
        self.disabled_for_frame = False
    def setup(self, reference):
        if reference.ndim == 3 and reference.shape[1:] == (1, 3):
            reference = reference[:, 0, :]
        if reference.ndim != 2 or reference.shape[1] != 3:
            raise ValueError('Log-scale must have shape [N, 3]')
        self.disabled_for_frame = False
        super().setup(reference[:, None, :])
    def bind(self, owner, node_count, allowed_nodes, optimizer, *, live=False):
        super().bind(owner, node_count, allowed_nodes, optimizer, live=live)
        optimizer.param_groups[-1]['name'] = 'early_regional_logscale'
    def _group(self):
        return next(g for g in self._optimizer.param_groups if g.get('name') == 'early_regional_logscale')
    def forward(self, training=False):
        return super().forward(training=training)[:, 0, :]
    def canonical_features(self):
        return super().canonical_features()[:, 0, :]
    def point_symbols(self):
        return self.symbols()[self.owner] if self.ready else None
    @torch.no_grad()
    def export(self, path, frame_index):
        if not self.ready:
            return None
        zeros = self.reference.detach().cpu().new_zeros(self.node_count * 3, 1)
        body = encode_opacity_delta(zeros, self.symbols().cpu().reshape(-1, 1),
            frame_index=frame_index, step=self.step, mode='logit')
        if body is None:
            return None
        owner = self.owner.cpu()
        packet = MAGIC + _regional_digest(self.reference.cpu(), owner, self.node_count) + body
        decoded = decode_regional_logscale(packet, self.reference[:, 0, :], frame_index=frame_index,
                                           owner=owner, node_count=self.node_count)
        if not torch.equal(decoded.view(torch.int32), self.canonical_features().view(torch.int32)):
            raise ValueError('Regional scale differs from actual packet')
        Path(path).write_bytes(packet)
        return dict(path=str(path), archive_bytes=len(packet), format=MAGIC.decode(),
                    active_regions=int(self.symbols().ne(0).any(-1).sum()))

def decode_regional_logscale(packet, reference, *, frame_index, owner, node_count):
    reference = torch.as_tensor(reference).detach().cpu().contiguous()
    if reference.ndim != 2 or reference.shape[1] != 3 or packet[:4] != MAGIC:
        raise ValueError('Invalid regional scale packet or reference')
    features = decode_feature_dc_delta(REGIONAL_MAGIC + packet[4:], reference[:, None, :],
        frame_index=frame_index, owner=owner, node_count=node_count)
    return features[:, 0, :].contiguous()
