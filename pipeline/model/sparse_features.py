"""Sparse, quantized base-color updates with an explicit feature stream tag."""
from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
import struct

import torch
from torch import nn

from .legacy_opacity_codec import decode_opacity_delta
from .sparse_updates import OpacityUpdateConfig, QuantizedOpacityUpdate, encode_opacity_delta
from .legacy_opacity_codec import HEADER, _step


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
    post_motion_only: bool = False
    tail_learning_rate_scale: float = 1.

    def scalar_config(self):
        return OpacityUpdateConfig(enabled=self.enabled, step=self.step,
            learning_rate=self.learning_rate, l1_weight=self.l1_weight,
            proximal_regularization=True, decay_with_motion=self.decay_with_motion,
            joint_with_birth=self.enabled and self.joint_with_birth, mode='logit', prune_threshold=0.)

    def validate(self):
        if not isinstance(self.post_motion_only, bool):
            raise ValueError('feature_dc_update.post_motion_only must be boolean')
        if not isinstance(self.gate_joint, bool):
            raise ValueError('feature_dc_update.gate_joint must be boolean')
        scale = self.tail_learning_rate_scale
        if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not 0 < scale <= 1:
            raise ValueError('feature_dc_update.tail_learning_rate_scale must be in (0, 1]')
        if self.gate_joint and self.sharing != 'spatial_region':
            raise ValueError('Joint gate colors require spatial region sharing')
        if self.sharing not in ('per_gs', 'spatial_region'):
            raise ValueError('Unknown DC feature sharing mode')
        if self.post_motion_only and (not self.enabled or self.sharing != 'spatial_region' or
                                     self.gate_joint or not self.joint_with_birth):
            raise ValueError('Post-motion regional DC requires enabled spatial sharing, frozen gates and tail fitting')
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


class _RegionalDCAdd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, reference, delta):
        return _merge_regional_dc(reference, delta)

    @staticmethod
    def backward(ctx, gradient):
        # Zero symbols preserve reference bits without killing their STE.
        return gradient, gradient


class _RegionalDCQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, step):
        return torch.round(value/step)*step

    @staticmethod
    def backward(ctx, gradient):
        return gradient, None


class RegionalFeatureDCUpdate(nn.Module):
    """Quantized regional increments, optionally following the live spatial cut."""
    def __init__(self, config, motion=None):
        super().__init__()
        config.validate()
        self.config = config
        # Motion is owned by the parent module, not a child of this module.
        object.__setattr__(self, 'motion', motion)
        self.step = _step(config.step)
        self.reference = self.owner = self.allowed = self.delta = None
        self.node_count = 0
        self.ready = False
        self._optimizer = None
        self._allowed_rows = self._allowed_scalar_indices = self._eligible_regions = None
        self._allowed_count = 0
        self.live = False

    @property
    def regional(self):
        return True

    def setup(self, reference):
        if reference.ndim != 3 or reference.shape[1:] != (1, 3):
            raise ValueError('DC features must have shape [N, 1, 3]')
        self.reference = reference.detach().clone()
        self.owner = self.allowed = self.delta = None
        self.node_count = 0
        self.ready = False
        self._optimizer = None
        self._allowed_rows = self._allowed_scalar_indices = self._eligible_regions = None
        self._allowed_count = 0
        self.live = False

    def bind(self, owner, node_count, allowed_nodes, optimizer, *, live=False):
        if self.ready:
            raise ValueError('Regional DC must bind exactly once after step200')
        owner = _regional_owner(owner, node_count, len(self.reference))
        allowed = torch.zeros(node_count, dtype=torch.bool, device=self.reference.device)
        nodes = torch.as_tensor(allowed_nodes, dtype=torch.long, device=allowed.device)
        if bool(((nodes < 0) | (nodes >= node_count)).any()):
            raise ValueError('Regional DC candidates lie outside the frozen tree')
        allowed[nodes] = True
        occupied = torch.bincount(owner, minlength=node_count).gt(0).to(allowed.device)
        if not live and bool((allowed & ~occupied).any()):
            raise ValueError('Regional DC candidates must be occupied gate owners')
        self.owner = owner.to(self.reference.device)
        self.node_count, self.allowed = node_count, allowed
        self._allowed_rows = allowed.nonzero(as_tuple=True)[0]
        self._allowed_count = self._allowed_rows.numel()
        self._allowed_scalar_indices = (self._allowed_rows[:, None] * 3 +
            torch.arange(3, device=allowed.device)[None, :]).flatten()
        self._eligible_regions = allowed.sum()
        self.delta = nn.Parameter(self.reference.new_zeros(node_count, 3))
        optimizer.add_param_group(dict(params=[self.delta], name='early_regional_dc',
            lr=self.config.learning_rate, betas=(.9, .999), eps=1e-8))
        self._optimizer, self.ready, self.live = optimizer, True, bool(live)

    def bind_live(self, motion, optimizer):
        if self.config.post_motion_only or motion is None:
            raise ValueError('Live regional DC requires an initialized motion tree')
        object.__setattr__(self, 'motion', motion)
        self.bind(motion.owner, len(motion.parents),
                  motion.node_counts.gt(0).nonzero(as_tuple=True)[0], optimizer, live=True)

    @torch.no_grad()
    def sync_owner(self):
        if self.ready and self.live:
            self.owner = self.motion.owner.detach().clone()

    @torch.no_grad()
    def freeze_owner(self):
        if not self.ready or not self.live:
            return
        self.sync_owner()
        self.allowed = torch.bincount(self.owner, minlength=self.node_count).gt(0)
        self._allowed_rows = self.allowed.nonzero(as_tuple=True)[0]
        self._allowed_count = self._allowed_rows.numel()
        self._allowed_scalar_indices = (self._allowed_rows[:, None] * 3 +
            torch.arange(3, device=self.allowed.device)[None, :]).flatten()
        self._eligible_regions = self.allowed.sum()
        self.delta.masked_fill_(~self.allowed[:, None], 0)
        self.live = False

    def prepare_step(self, step, final_ratio):
        if not self.ready:
            return
        start = 201 if self.config.post_motion_only else 1
        self.delta.requires_grad_(step >= start)
        progress = min(max((step-start)/(300.-start), 0.), 1.)
        factor = final_ratio ** progress if self.config.decay_with_motion else 1.
        if step > 200:
            factor *= self.config.tail_learning_rate_scale
        self._group()['lr'] = self.config.learning_rate * factor

    def _group(self):
        return next(g for g in self._optimizer.param_groups if g.get('name') == 'early_regional_dc')

    def forward(self, training=False):
        if not self.ready:
            return self.reference
        quantized = _RegionalDCQuantize.apply(self.delta, self.step)
        quantized = quantized * self.allowed[:, None]
        if self.live and training and self.motion.shared_discovery_active:
            weights = self.motion._sampled_path_weights(sample_if_needed=False)
            if not self.config.gate_joint:
                weights = weights.detach()
            delta = (weights[..., None] * quantized[self.motion.spatial_paths]).sum(1)
        else:
            owner = self.motion.owner if self.live else self.owner
            delta = quantized[owner]
        return _RegionalDCAdd.apply(self.reference, delta[:, None, :])

    def symbols(self):
        if not self.ready:
            return None
        return torch.round(self.delta.detach()/self.step) * self.allowed[:, None]

    def canonical_features(self):
        reference = self.reference.detach().cpu()
        if not self.ready:
            return reference
        delta = self.symbols().cpu().to(reference) * self.step
        return _merge_regional_dc(reference, delta[self.owner.cpu(), None, :])

    def _row_costs(self, symbols=None):
        # Exact uncompressed SDR1 scalar-row bytes, assigned to their regions.
        costs = self.delta.new_zeros(self.node_count)
        if self._allowed_count:
            if symbols is None:
                values = self.delta.detach().index_select(0, self._allowed_rows)
                symbols = torch.round(values / self.step)
            active = symbols.ne(0).flatten()
            indices = self._allowed_scalar_indices
            # The preceding active scalar determines the encoded varint gap;
            # fixed-size scans avoid a host synchronization on each step.
            previous = torch.cummax(torch.where(active, indices, indices.new_full((), -1)), 0).values
            previous = torch.cat((indices.new_full((1,), -1), previous[:-1]))
            gaps = indices - previous
            lengths = 1 + gaps.ge(128).long() + gaps.ge(16384).long()
            compact = ((lengths + 2) * active).reshape(-1, 3).sum(-1).to(costs)
            costs.index_copy_(0, self._allowed_rows, compact)
        return costs

    @torch.no_grad()
    def rate_terms(self, byte_price, width=1.):
        zero = self.reference.new_zeros(())
        if not self.ready:
            return dict(loss=zero, raw_bytes_surrogate=zero, l1_loss=zero,
                        active_rows=zero, active_regions=zero, eligible_regions=zero)
        values = self.delta.index_select(0, self._allowed_rows)
        symbols = torch.round(values / self.step)
        active = symbols.ne(0).any(-1)
        raw = self._row_costs(symbols).sum() + active.any().to(self.delta)*(len(REGIONAL_MAGIC)+32+HEADER.size)
        l1 = self.config.l1_weight*values.abs().mean() if values.numel() else zero
        return dict(loss=raw*byte_price+l1, raw_bytes_surrogate=raw, l1_loss=l1,
            active_rows=symbols.ne(0).sum(), active_regions=active.sum(), eligible_regions=self._eligible_regions)

    @torch.no_grad()
    def after_step(self, byte_price, *, proximal):
        if not self.ready:
            return
        self.delta.clamp_(-32767*self.step, 32767*self.step)
        self.delta.masked_fill_(~self.allowed[:, None], 0)
        state = self._optimizer.state.get(self.delta, {})
        if 'exp_avg_sq' not in state:
            return
        group = self._group()
        lr = group['lr']
        metric = (state['exp_avg_sq']/(1-group['betas'][1]**int(state['step']))).sqrt().add(group['eps'])
        if self.config.l1_weight and self._allowed_count:
            threshold = lr*self.config.l1_weight/(self._allowed_count*3)/metric
            self.delta.copy_(self.delta.sign()*(self.delta.abs()-threshold).clamp_min(0))
        if proximal and byte_price > 0:
            gain = (.5*metric*self.delta.square()/lr).sum(-1)
            active = self.symbols().ne(0).any(-1)
            self.delta.masked_fill_((active & gain.lt(self._row_costs()*byte_price))[:, None], 0)
        self.delta.masked_fill_(~self.allowed[:, None], 0)

    def finish(self):
        if self.ready:
            self.delta.requires_grad_(False)

    def capture(self):
        if self.reference is None:
            return None
        result = dict(config=asdict(self.config), reference=self.reference.detach().clone(), ready=self.ready)
        if self.ready:
            result.update(owner=self.owner.detach().clone(), node_count=self.node_count, live=self.live,
                allowed=self.allowed.detach().clone(), delta=self.delta.detach().clone(),
                trainable=self.delta.requires_grad,
                hyper={k:v for k,v in self._group().items() if k!='params'},
                state={k:v.detach().clone() if torch.is_tensor(v) else v for k,v in
                       self._optimizer.state.get(self.delta, {}).items()})
        return result

    def restore(self, packet, device, optimizer):
        if packet is None or {**asdict(FeatureDCUpdateConfig()), **packet['config']} != asdict(self.config):
            raise ValueError('Regional DC checkpoint configuration differs or is missing')
        self.setup(packet['reference'].to(device))
        if not packet['ready']:
            return
        self.bind(packet['owner'], int(packet['node_count']),
                  packet['allowed'].nonzero(as_tuple=True)[0], optimizer,
                  live=bool(packet.get('live', False)))
        if packet['delta'].shape != self.delta.shape or not bool(torch.isfinite(packet['delta']).all()):
            raise ValueError('Regional DC checkpoint lost its node layout')
        self.delta.data.copy_(packet['delta'].to(device))
        if bool(self.delta[~self.allowed].ne(0).any()):
            raise ValueError('Regional DC checkpoint changed an unsupported region')
        self.delta.requires_grad_(bool(packet['trainable']))
        self._group().update(packet['hyper'])
        optimizer.state[self.delta] = {k:v.to('cpu' if k=='step' else device) if torch.is_tensor(v) else v
                                      for k,v in packet['state'].items()}

    @torch.no_grad()
    def export(self, path, frame_index):
        if not self.ready:
            return None
        zeros = self.reference.detach().cpu().new_zeros(self.node_count*3, 1)
        body = encode_opacity_delta(zeros, self.symbols().cpu().reshape(-1, 1),
                                   frame_index=frame_index, step=self.step, mode='logit')
        if body is None:
            return None
        reference, owner = self.reference.detach().cpu(), self.owner.cpu()
        packet = REGIONAL_MAGIC+_regional_digest(reference, owner, self.node_count)+body
        decoded = decode_feature_dc_delta(packet, reference, frame_index=frame_index,
                                         owner=owner, node_count=self.node_count)
        if not torch.equal(decoded.view(torch.int32), self.canonical_features().view(torch.int32)):
            raise ValueError('Regional DC differs from its actual packet')
        path = Path(path)
        path.write_bytes(packet)
        return dict(path=str(path), archive_bytes=len(packet), format=REGIONAL_MAGIC.decode(),
            channels=3, sharing='spatial_region', timing='before_births',
            start_step=201 if self.config.post_motion_only else 1,
            active_regions=int(self.symbols().ne(0).any(-1).sum()), eligible_regions=int(self.allowed.sum()))


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
