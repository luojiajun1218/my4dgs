"""Quantized old-opacity deltas and differentiable sparse block prices."""

from dataclasses import dataclass, asdict
import math
from pathlib import Path
import zlib
import struct

import torch
from torch import nn

from .legacy_opacity_codec import (HEADER, MAGIC, _step, _varint,
    opacity_reference_hash, decode_opacity_delta, MODES,
    canonical_alpha, canonical_alpha_logits)
from .sparse_integer_codec import OPACITY_MAGICS, pack_layout, encode_opacity_layout


def sparse_block_support(value, step, *, continuous=None, differentiable=False, width=1.):
    """True nonzero block in forward; bin-scaled smooth L0 gradient backward."""
    hard = torch.round(value / step).ne(0).any(-1).to(value.dtype)
    if not differentiable:
        return hard
    continuous = value if continuous is None else continuous
    bins = continuous / step
    # Evaluate the sparse derivative on the transmitted bins. In a zero bin
    # abs has zero subgradient: a large Adam step must not turn an already-free
    # sub-bin update into an opposite-sign nonzero update solely due to rate.
    bins = bins + (torch.round(value / step) - bins).detach()
    soft = -torch.expm1(-bins.abs().sum(-1) / width)
    return hard + (soft - soft.detach())


@dataclass
class OpacityUpdateConfig:
    enabled: bool = False
    step: float = .05
    learning_rate: float = .05
    prune_threshold: float = .01
    proximal_regularization: bool = False
    mode: str = 'logit'
    l1_weight: float = 0.
    decay_with_motion: bool = False

    def validate(self):
        if not isinstance(self.enabled, bool):
            raise ValueError('opacity_update.enabled must be boolean')
        if not isinstance(self.proximal_regularization, bool):
            raise ValueError('opacity_update.proximal_regularization must be boolean')
        if not isinstance(self.decay_with_motion, bool):
            raise ValueError('opacity_update.decay_with_motion must be boolean')
        if (isinstance(self.l1_weight, bool) or not math.isfinite(self.l1_weight) or self.l1_weight < 0):
            raise ValueError('opacity_update.l1_weight must be finite and nonnegative')
        _step(self.step)
        if self.mode not in ('logit', 'alpha'):
            raise ValueError('Unknown opacity update mode')
        if (isinstance(self.learning_rate, bool) or not math.isfinite(self.learning_rate) or
                self.learning_rate <= 0):
            raise ValueError('opacity_update.learning_rate must be positive')
        if (isinstance(self.prune_threshold, bool) or not math.isfinite(self.prune_threshold) or
                not 0 <= self.prune_threshold < 1):
            raise ValueError('opacity_update.prune_threshold must be in [0, 1)')


class _QuantizedLogits(torch.autograd.Function):
    @staticmethod
    def forward(ctx, reference, delta, step):
        symbols = torch.round(delta / step)
        # Keep raw reference bits for zero symbols, including signed zero.
        return torch.where(symbols.ne(0), reference + symbols * step, reference)

    @staticmethod
    def backward(ctx, gradient):
        return None, gradient, None


class _QuantizedAlpha(torch.autograd.Function):
    @staticmethod
    def forward(ctx, canonical_reference, render_reference, delta, step):
        symbols = torch.round(delta / step)
        target = (canonical_reference + symbols * step).clamp(0.,1.)
        return torch.where(symbols.ne(0),target,render_reference)

    @staticmethod
    def backward(ctx, gradient):
        return None,None,gradient,None


class _PreserveZeroLogits(torch.autograd.Function):
    @staticmethod
    def forward(ctx, reference, logits, zero):
        return torch.where(zero,reference,logits)

    @staticmethod
    def backward(ctx, gradient):
        return None,gradient,None


def encode_opacity_delta(reference, symbols, *, frame_index, step, mode='logit'):
    """Write exact quantized opacity deltas using the shortest lossless layout."""
    reference = torch.as_tensor(reference).detach().cpu().contiguous()
    step = _step(step)
    values = torch.as_tensor(symbols).detach().cpu()
    if (values.shape != reference.shape or not bool(torch.isfinite(values).all()) or
            not torch.equal(values, values.round()) or bool(values.abs().gt(32767).any())):
        raise ValueError('Opacity symbols must be aligned signed-int16 integers')
    if isinstance(frame_index, bool) or not isinstance(frame_index, int) or not 1 <= frame_index <= 0xffffffff:
        raise ValueError('Opacity frame must be a positive uint32')
    prior = opacity_reference_hash(reference)
    if mode not in ('logit','alpha'):
        raise ValueError('Unknown opacity update mode')
    magic = MODES['alpha_canonical' if mode == 'alpha' else mode]
    rows = values[:, 0].ne(0).nonzero(as_tuple=True)[0]
    if not len(rows):
        return None
    body = bytearray()
    previous = -1
    for row in rows.tolist():
        body.extend(_varint(row - previous))
        body.extend(struct.pack('<h', int(values[row, 0])))
        previous = row
    prefix = HEADER.pack(magic, frame_index, len(reference), len(rows), step,
                         len(body), 0, prior)
    checksum = zlib.crc32(prefix + body)
    header = HEADER.pack(magic, frame_index, len(reference), len(rows), step,
                         len(body), checksum, prior)
    packet = header + zlib.compress(body, 6)
    layout = encode_opacity_layout(header + body)
    packet = min((packet, pack_layout(layout, HEADER.size + len(body), OPACITY_MAGICS)), key=len)
    # The existing decoder is also the byte-exact definition of materialization.
    decode_opacity_delta(packet, reference, frame_index=frame_index)
    return packet


class QuantizedOpacityUpdate(nn.Module):
    """Fit old logits relative to the preceding decoded frame; freeze at birth."""
    def __init__(self, config):
        super().__init__()
        config.validate()
        self.config = config
        self.step = _step(config.step)
        self.reference = None
        self.delta = None
        self.rows = None
        self.optimizer = None
        self.committed = False

    def setup(self, reference):
        self.reference = reference.detach().clone()
        self.alpha_reference = (canonical_alpha(self.reference) if self.config.mode=='alpha' else None)
        self.render_alpha_reference = (self.reference.sigmoid() if self.config.mode=='alpha' else None)
        self.delta = nn.Parameter(torch.zeros_like(reference))
        self.rows = torch.arange(len(reference), device=reference.device)
        self.optimizer = torch.optim.Adam([self.delta], lr=self.config.learning_rate, eps=1e-8)
        self.committed = False

    def forward(self):
        if self.config.mode=='alpha':
            alpha=self.alphas()
            logits=torch.logit(alpha.clamp(1e-6,1-1e-6))
            logits=torch.where(alpha.le(0),logits.new_full((),-100.),
                torch.where(alpha.ge(1),logits.new_full((),100.),logits))
            zero=self.symbols()[self.rows].eq(0)
            return _PreserveZeroLogits.apply(self.reference[self.rows],logits,zero)
        return _QuantizedLogits.apply(self.reference, self.delta, self.step)[self.rows]

    def alphas(self):
        if self.config.mode=='alpha':
            return _QuantizedAlpha.apply(self.alpha_reference,self.render_alpha_reference,
                self.delta,self.step)[self.rows]
        return self().sigmoid()

    def rate_terms(self, byte_price, width=1.):
        value = self.delta[self.rows]
        if self.config.proximal_regularization:
            value = value.detach()
        support = sparse_block_support(value, self.step, differentiable=True, width=width)
        nonzero = torch.round(value / self.step).ne(0).sum()
        # Gap + int16 is three raw bytes for a typical row. Actual gap lengths
        # and compressed bytes are checked at export, not assumed to be equal.
        raw_bytes = 3 * support.sum() + HEADER.size * nonzero.gt(0).to(value)
        l1 = self.config.l1_weight * value.abs().mean()
        return dict(loss=raw_bytes * byte_price + l1, l1_loss=l1,
                    raw_bytes_surrogate=raw_bytes,
                    active_rows=nonzero)

    @torch.no_grad()
    def optimizer_step(self, byte_price, proximal=False, lr_factor=1.):
        learning_rate = self.config.learning_rate * lr_factor
        self.optimizer.param_groups[0]['lr'] = learning_rate
        self.optimizer.step()
        self.delta.clamp_(min=-32767 * self.step, max=32767 * self.step)
        if self.config.mode=='alpha':
            self.delta.copy_(torch.minimum(torch.maximum(self.delta,-self.alpha_reference),1-self.alpha_reference))
        if ((proximal and byte_price > 0) or
                (self.config.proximal_regularization and self.config.l1_weight > 0)):
            state = self.optimizer.state.get(self.delta, {})
            if 'exp_avg_sq' in state:
                t = int(state['step'])
                metric = (state['exp_avg_sq'] / (1 - .999 ** t)).sqrt().add(1e-8)
                if self.config.proximal_regularization and self.config.l1_weight:
                    threshold = learning_rate * self.config.l1_weight / self.delta.numel() / metric
                    self.delta.copy_(self.delta.sign() * (self.delta.abs() - threshold).clamp_min(0))
                if proximal and byte_price > 0:
                    gain = .5 * metric * self.delta.square() / learning_rate
                    active = torch.round(self.delta / self.step).ne(0)
                    prune = gain.lt(3 * byte_price)
                    if not self.config.proximal_regularization:
                        prune &= active
                    self.delta[prune] = 0
        self.optimizer.zero_grad(set_to_none=True)

    def symbols(self):
        return torch.round(self.delta.detach() / self.step)

    @torch.no_grad()
    def canonical_logits(self, full=False):
        reference = self.reference.detach().cpu()
        symbols = self.symbols().cpu()
        if self.config.mode=='alpha':
            result=canonical_alpha_logits(reference,symbols,self.step)
            return result if full else result[self.rows.cpu()]
        result = reference.clone()
        active = symbols[:, 0].ne(0)
        result[active, 0] = reference[active, 0] + symbols[active, 0] * self.step
        return result if full else result[self.rows.cpu()]

    def commit(self, rows):
        self.rows = rows.detach().to(self.reference.device, dtype=torch.long).clone()
        self.committed = True
        self.delta.requires_grad_(False)

    @torch.no_grad()
    def export(self, path, frame_index):
        reference = self.reference[self.rows].cpu()
        symbols = self.symbols()[self.rows].cpu()
        packet = encode_opacity_delta(reference, symbols, frame_index=frame_index, step=self.step, mode=self.config.mode)
        if packet is None:
            return None
        decoded = decode_opacity_delta(packet, reference, frame_index=frame_index)
        if not torch.equal(decoded.logits, self.canonical_logits()):
            raise ValueError('Opacity packet differs from materialized quantized logits')
        path = Path(path)
        path.write_bytes(packet)
        return dict(path=str(path), archive_bytes=len(packet), rows=len(decoded.rows),
                    step=self.step, mode=self.config.mode, timing='before_births',
                    format=packet[:4].decode('ascii'),
                    raw_body_bytes=sum(len(_varint(int(gap))) + 2 for gap in
                        torch.diff(torch.cat((decoded.rows.new_tensor([-1]), decoded.rows))).tolist()))

    def capture(self):
        if self.reference is None:
            return None
        return dict(config=asdict(self.config), reference=self.reference,
                    delta=self.delta.detach(), rows=self.rows,
                    committed=self.committed, optimizer=self.optimizer.state_dict())

    def restore(self, packet, device):
        normalized = None if packet is None else {**asdict(OpacityUpdateConfig()), **packet['config']}
        if packet is None or normalized != asdict(self.config):
            raise ValueError('Opacity checkpoint configuration differs or is missing')
        self.setup(packet['reference'].to(device))
        self.delta.data.copy_(packet['delta'].to(device))
        self.rows = packet['rows'].to(device)
        self.committed = bool(packet['committed'])
        self.delta.requires_grad_(not self.committed)
        self.optimizer.load_state_dict(packet['optimizer'])
