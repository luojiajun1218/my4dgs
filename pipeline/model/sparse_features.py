"""Sparse, quantized base-color updates with an explicit feature stream tag."""
from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
import struct
import weakref

import torch
from torch import nn

from .legacy_opacity_codec import HEADER, decode_opacity_delta
from .sparse_updates import OpacityUpdateConfig, QuantizedOpacityUpdate, _QuantizedLogits
from .spatial_gate import spatial_accumulate


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
    """Quantize GS or region RGB increments using the scalar additive coder."""
    def __init__(self, config, motion=None):
        super().__init__()
        config.validate()
        self.config = config
        self.scalar = QuantizedOpacityUpdate(config.scalar_config())
        self._motion_ref = None if motion is None else weakref.ref(motion)
        self.reference = None
        self._relative_active = False

    @property
    def regional(self):
        return self.config.sharing == 'spatial_region'

    def _motion(self):
        motion = None if self._motion_ref is None else self._motion_ref()
        if motion is None or not hasattr(motion, 'parents'):
            raise ValueError('Regional DC requires a prepared spatial motion model')
        if self.config.gate_joint and not (motion.stochastic_image_gates and motion.joint_image_gradients):
            raise ValueError('Joint regional DC requires stochastic joint image gates')
        return motion

    def _ready(self):
        return not self.regional or self._motion()._hard_gate_frozen or self.config.gate_joint

    def setup(self, reference):
        if reference.ndim != 3 or reference.shape[1:] != (1, 3):
            raise ValueError('DC features must have shape [N, 1, 3]')
        if self.regional:
            motion = self._motion()
            if len(reference) != len(motion.owner) or reference.dtype != torch.float32:
                raise ValueError('Regional DC reference must match float32 old GS attributes')
            self.reference = reference.detach().clone()
            self.scalar.setup(reference.new_zeros(len(motion.parents) * 3, 1))
            self._relative_active = self.config.gate_joint and motion.relative_regions
        else:
            self.scalar.setup(reference.reshape(-1, 1))

    def _regional_table(self, *, canonical=False):
        if not self._relative_active:
            return (self.scalar.canonical_logits() if canonical else self.scalar()).reshape(-1, 3)
        local = self.scalar.delta.reshape(-1, 3)
        if canonical:
            symbols = self.scalar.symbols().detach().cpu().reshape(-1, 3).to(local.dtype)
        else:
            normalized = local / self.scalar.step
            symbols = normalized.round().detach() + (normalized - normalized.detach())
        symbols = torch.cat((torch.zeros_like(symbols[:1]), symbols[1:]))
        parents = self._motion().parents.to(symbols.device)
        return spatial_accumulate(symbols, parents) * self.scalar.step

    def forward(self, training=False):
        if self.regional:
            if not self._ready():
                return self.reference
            motion = self._motion()
            if self.config.gate_joint and training and not motion._hard_gate_frozen:
                weights = motion._sampled_path_weights(sample_if_needed=False)
                table = self._regional_table()
                delta = (weights[..., None] * table[motion.spatial_paths]).sum(1)[:, None]
            else:
                delta = (self._regional_table().reshape(-1, 1, 3)[motion.owner]
                         if self._relative_active else
                         self.scalar.delta.reshape(-1, 1, 3)[motion.owner])
            return _QuantizedLogits.apply(self.reference, delta, self.scalar.step)
        return self.scalar().reshape(-1, 1, 3)

    def canonical_features(self):
        if self.regional:
            reference = self.reference.detach().cpu()
            if not self._ready():
                return reference
            delta = self._regional_table(canonical=True).reshape(-1, 1, 3)[self._motion().owner.cpu()]
            return _merge_regional_dc(reference, delta)
        return self.scalar.canonical_logits().reshape(-1, 1, 3)

    def rate_terms(self, byte_price, width=1.):
        if self.config.gate_joint:
            return self._joint_rate_terms(byte_price)
        terms = self.scalar.rate_terms(byte_price, width)
        if not self._ready():
            return {name: value * 0 for name, value in terms.items()}
        # One explicit field tag is also present in the actual archive.
        tag = terms['active_rows'].gt(0).to(terms['loss']) * (36 if self.regional else len(MAGIC))
        return {**terms, 'loss': terms['loss'] + tag * byte_price,
                'raw_bytes_surrogate': terms['raw_bytes_surrogate'] + tag}

    def _selected_regions(self):
        motion = self._motion()
        return torch.bincount(motion.owner, minlength=len(motion.parents)).gt(0)

    def _joint_rate_terms(self, byte_price):
        """Price selected color rows; STE probabilities train only the gates.

        Three raw bytes per scalar is the existing scalar coder's typical-row
        proxy. Final compressed archive sizes are measured independently.
        Color parameters retain photo Adam plus the existing scalar proximal.
        """
        motion = self._motion()
        delta = self.scalar.delta.detach().reshape(-1, 3)
        if self._relative_active:
            with torch.no_grad():
                table = self._regional_table().detach()
            channels = table.ne(0).sum(-1).to(delta)
        else:
            table = delta
            channels = self.scalar.symbols().reshape(-1, 3).ne(0).sum(-1).to(delta)
        selected = self._selected_regions().to(delta)
        probabilities = (selected if motion._hard_gate_frozen else
                         motion._region_selection_probabilities())
        hard_rows = (selected * channels).sum()
        soft_rows = (probabilities * channels).sum()
        raw = 3 * (hard_rows + (soft_rows - soft_rows.detach()))
        raw = raw + hard_rows.gt(0).to(raw) * (HEADER.size + 36)
        per_node_l1 = self.config.l1_weight * table.abs().sum(-1) / delta.numel()
        hard_l1 = (selected * per_node_l1).sum()
        soft_l1 = (probabilities * per_node_l1).sum()
        l1 = hard_l1 + (soft_l1 - soft_l1.detach())
        return dict(loss=raw * byte_price + l1, l1_loss=l1,
                    raw_bytes_surrogate=raw, active_rows=hard_rows)

    @torch.no_grad()
    def _mask_unused_regions(self):
        used = self._selected_regions()
        self.scalar.delta.reshape(-1, 3)[~used] = 0
        if self.scalar.delta.grad is not None:
            self.scalar.delta.grad.reshape(-1, 3)[~used] = 0
        state = self.scalar.optimizer.state.get(self.scalar.delta, {})
        for key in ('exp_avg', 'exp_avg_sq'):
            if key in state:
                state[key].reshape(-1, 3)[~used] = 0

    def optimizer_step(self, *args, **kwargs):
        if not self._ready():
            self.scalar.optimizer.zero_grad(set_to_none=True)
            return
        if self._relative_active and self._motion()._hard_gate_frozen:
            # Finish the last gradient in its original local coordinates,
            # then change coordinates without changing quantized colors.
            self.scalar.optimizer_step(*args, **kwargs)
            self._finalize_relative_regions()
            self._mask_unused_regions()
            return
        if self.config.gate_joint and self._motion()._hard_gate_frozen:
            self._mask_unused_regions()
        self.scalar.optimizer_step(*args, **kwargs)

    @torch.no_grad()
    def _finalize_relative_regions(self):
        if not self._relative_active:
            return
        if not self._motion()._hard_gate_frozen:
            raise ValueError('Freeze gate before materializing inherited regional colors')
        table = self._regional_table(canonical=True)
        if bool(torch.round(table / self.scalar.step).abs().gt(32767).any()):
            raise ValueError('Accumulated regional color exceeds the scalar codec range')
        self.scalar.delta.copy_(table.reshape_as(self.scalar.delta))
        self.scalar.optimizer.state.pop(self.scalar.delta, None)
        self._relative_active = False

    def commit(self, rows):
        if self.regional:
            self._finalize_relative_regions()
            self.reference = self.reference[rows].clone()
            motion = self._motion()
            if len(self.reference) != len(motion.owner):
                raise ValueError('Regional DC retirement must follow motion retirement')
            self._mask_unused_regions()
            self.scalar.commit(torch.arange(len(self.scalar.reference), device=rows.device))
            return
        channels = torch.arange(3, device=rows.device)
        self.scalar.commit((rows[:, None] * 3 + channels).flatten())

    def finish(self):
        self.scalar.finish()

    def capture(self):
        if self.scalar.reference is None:
            return None
        packet = dict(config=asdict(self.config), scalar=self.scalar.capture())
        if self.regional:
            packet['regional_reference'] = self.reference
            if self.config.gate_joint and self._motion().relative_regions:
                packet['regional_relative_active'] = self._relative_active
        return packet

    def restore(self, packet, device):
        normalized = None if packet is None else {**asdict(FeatureDCUpdateConfig()), **packet['config']}
        if packet is None or normalized != asdict(self.config):
            raise ValueError('DC feature checkpoint configuration differs or is missing')
        self.scalar.restore(packet['scalar'], device)
        if self.regional:
            motion = self._motion()
            relative = packet.get('regional_relative_active', False)
            configured = self.config.gate_joint and motion.relative_regions
            if (not isinstance(relative, bool) or (relative and not configured) or
                    (configured and 'regional_relative_active' not in packet)):
                raise ValueError('Regional color checkpoint lacks its inherited coordinate state')
            self._relative_active = relative
            reference = packet.get('regional_reference')
            if (not torch.is_tensor(reference) or reference.shape != (len(motion.owner), 1, 3) or
                    reference.dtype != torch.float32 or not bool(torch.isfinite(reference).all()) or
                    self.scalar.reference.shape != (len(motion.parents) * 3, 1) or
                    bool(self.scalar.reference.ne(0).any()) or
                    not torch.equal(self.scalar.rows, torch.arange(len(self.scalar.reference), device=device))):
                raise ValueError('Regional DC checkpoint differs from decoded motion layout')
            self.reference = reference.to(device).clone()
            return
        if len(self.scalar.rows) % 3:
            raise ValueError('Incomplete RGB channel rows')
        rows = self.scalar.rows.reshape(-1, 3)
        if not torch.equal(rows, (rows[:, :1] // 3) * 3 + torch.arange(3, device=device)):
            raise ValueError('DC feature rows lost GS/channel alignment')

    @torch.no_grad()
    def export(self, path, frame_index):
        # The scalar export validates exact values against its decoder first.
        path = Path(path)
        if not self._ready():
            return None
        if self.regional:
            if not self._motion()._hard_gate_frozen:
                return None
            self._finalize_relative_regions()
            self._mask_unused_regions()
        info = self.scalar.export(path, frame_index)
        if info is None:
            return None
        if self.regional:
            motion = self._motion()
            reference = self.reference.detach().cpu().contiguous()
            owner = _regional_owner(motion.owner, len(motion.parents), len(reference))
            packet = REGIONAL_MAGIC + _regional_digest(reference, owner, len(motion.parents)) + path.read_bytes()
            decoded = decode_feature_dc_delta(packet, reference, frame_index=frame_index,
                                              owner=owner, node_count=len(motion.parents))
        else:
            packet = MAGIC + path.read_bytes()
            reference = self.scalar.reference[self.scalar.rows].reshape(-1, 1, 3)
            decoded = decode_feature_dc_delta(packet, reference, frame_index=frame_index)
        if not torch.equal(decoded, self.canonical_features()):
            raise ValueError('DC feature packet differs from materialized features')
        path.write_bytes(packet)
        return {**info, 'archive_bytes': len(packet), 'format': packet[:4].decode(),
                'channels': 3, 'sharing': self.config.sharing, 'timing': 'before_births'}
