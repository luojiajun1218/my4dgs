"""Actual spatial packet lengths with a differentiable byte-context rate proxy.

The rate is periodically measured by the production automatic lossless encoder,
before births. Intervening values and backward use a first-order model of the
*same* planar varint streams, calibrated to that packet length. This is an STE estimate, not a
claim that zlib/XZ or discrete row selection has an exact derivative.
"""
import hashlib

import numpy as np
import torch

from .rigid_tree_codec import (CHECKSUM_BYTES, _sparse_write,
                               compress_spatial_packet, spatial_gate_payload,
                               spatial_gate_magic)
from .sparse_integer_codec import (encode_unsigned_values, _spatial_prefix,
                                   _Reader)


def _bytes(values):
    return np.frombuffer(encode_unsigned_values(values), dtype=np.uint8)


def _streams(t, r):
    t, r = np.asarray(t, dtype=np.int64), np.asarray(r, dtype=np.int64)
    at, ar = t.any(1), r.any(1)
    rows = np.flatnonzero(at | ar)
    streams = [_bytes(np.diff(np.r_[-1, rows])),
               (at[rows].astype(np.uint8) | (ar[rows].astype(np.uint8) << 1))]
    for values, active in ((t, at), (r, ar)):
        zigzag = np.where(values >= 0, 2 * values, -2 * values - 1)
        streams.extend(_bytes(zigzag[active, axis]) for axis in range(3))
    return streams


class MotionCodecRate:
    def __init__(self, empty_packet, regions, points, refresh_steps=20,
                 previous_opened=None):
        # The gate section can now be variable-length. Parse its boundary
        # rather than subtracting a full bitmap from the end of the packet.
        self.regions, self.points = int(regions), int(points)
        reader = _Reader(empty_packet)
        _, _, _, gate_start = _spatial_prefix(reader)
        self.prefix = bytes(empty_packet[:gate_start])
        self.previous_opened = (None if previous_opened is None else
                                torch.as_tensor(previous_opened).detach().cpu().bool().clone())
        self.refresh_steps = int(refresh_steps)
        self.last_refresh = -1
        self.tables = None
        self._device_tables = {}
        self.last_scale = 1.
        self.last_packet_bytes = 0
        self.last_entropy_bytes = 0.
        self.last_measure_step = -1

    def packet(self, rt, rr, gt, gr, opened):
        arrays = [np.asarray(value, dtype='<i2') for value in (rt, rr, gt, gr)]
        gates, delta = spatial_gate_payload(opened, self.previous_opened)
        body = bytearray(spatial_gate_magic(self.prefix[:4], delta) + self.prefix[4:] + gates)
        _sparse_write(body, arrays[0][1:], arrays[1][1:])
        _sparse_write(body, arrays[2], arrays[3])
        body.extend(hashlib.sha256(body).digest()[:CHECKSUM_BYTES])
        return compress_spatial_packet(bytes(body), opened, len(self.prefix), len(gates))

    def measure(self, rt, rr, gt, gr, opened, step):
        arrays = [np.asarray(value, dtype='<i2') for value in (rt, rr, gt, gr)]
        packet = self.packet(*arrays, opened)
        self.last_packet_bytes = len(packet)
        self.last_measure_step = int(step)
        streams = [_streams(arrays[0][1:], arrays[1][1:]), _streams(arrays[2], arrays[3])]
        if self.tables is None or (step != self.last_refresh and step % self.refresh_steps == 0):
            tables = []
            for section_streams in streams:
                section = []
                for stream in section_streams:
                    previous = np.r_[256, stream[:-1]] if len(stream) else np.empty(0, dtype=int)
                    counts = np.bincount(previous * 256 + stream,
                        minlength=257 * 256).reshape(257, 256).astype(np.float64)
                    marginal = (np.bincount(stream, minlength=256) + .5) / (len(stream) + 128.)
                    probability = (counts + 8. * marginal[None]) / (counts.sum(1, keepdims=True) + 8.)
                    section.append(-np.log2(probability).astype(np.float32))
                tables.append(np.stack(section))
            self.tables = torch.from_numpy(np.stack(tables))
            self._device_tables.clear()
            self.last_refresh = int(step)
        self.last_entropy_bytes = 0.
        for section, section_streams in enumerate(streams):
            for index, stream in enumerate(section_streams):
                if len(stream):
                    previous = np.r_[256, stream[:-1]]
                    self.last_entropy_bytes += float(self.tables[section, index].numpy()[previous, stream].sum()) / 8.
        return len(packet)

    def _book(self, section, like):
        key = (str(like.device), like.dtype)
        if key not in self._device_tables:
            self._device_tables[key] = self.tables.to(device=like.device, dtype=like.dtype)
        return self._device_tables[key][section]

    @staticmethod
    def _previous(active):
        rows = torch.arange(len(active), device=active.device)
        seen = torch.where(active, rows, -1).cummax(0).values
        previous = torch.cat((seen.new_tensor([-1]), seen[:-1]))
        return previous

    @staticmethod
    def _words(values, previous, tables, max_bytes=3):
        """Per-row byte-context cost, including within-varint transitions."""
        lengths = torch.ones_like(values)
        for shift in range(7, 7 * max_bytes, 7):
            lengths += values.ge(1 << shift)
        prior = values[previous.clamp_min(0)]
        prior_lengths = lengths[previous.clamp_min(0)]
        last = (prior >> (7 * (prior_lengths - 1))) & 127
        context = torch.where(previous[:, None].ge(0), last, 256)
        axes = torch.arange(values.shape[1], device=values.device)[None]
        result = tables.new_zeros(values.shape)
        for index in range(max_bytes):
            byte = ((values >> (7 * index)) & 127) | (values.ge(1 << (7 * (index + 1))).long() << 7)
            price = tables[axes, context, byte]
            result += torch.where(lengths.gt(index), price, 0.)
            context = byte
        return result.sum(-1)

    def costs(self, t, r, st, sr, section, allowed=None):
        """Estimated bytes per sparse row; hard symbols, soft block support."""
        if section == 0 and len(t) == self.regions:
            # Region row zero is the identity and is absent from the section.
            allowed = None if allowed is None else allowed[1:]
            return torch.cat((t.new_zeros(1), self.costs(t[1:], r[1:], st[1:], sr[1:], section, allowed)))
        book = self._book(section, t)
        it, ir = t.detach().round().long(), r.detach().round().long()
        at, ar = it.ne(0).any(-1), ir.ne(0).any(-1)
        if allowed is not None:
            at, ar = at & allowed, ar & allowed
            st, sr = st * allowed, sr * allowed
        active = at | ar
        previous = self._previous(active)
        rows = torch.arange(len(active), device=t.device)
        gaps = (rows - previous)[:, None]
        gap_cost = self._words(gaps, previous, book[:1], max_bytes=max(1, (len(active).bit_length() + 6) // 7))
        flags = at.long() + 2 * ar.long()
        prior_flags = torch.where(previous.ge(0), flags[previous.clamp_min(0)], 256)
        flag_cost = book[1, prior_flags, flags]
        result = (st + sr - st * sr) * (gap_cost + flag_cost)
        for values, hard, soft, tables in ((it, at, st, book[2:5]), (ir, ar, sr, book[5:8])):
            zigzag = torch.where(values.ge(0), 2 * values, -2 * values - 1)
            result = result + soft * self._words(zigzag, self._previous(hard), tables)
        return result / 8.

    def capture(self):
        return dict(prefix=self.prefix, previous_opened=self.previous_opened,
                    regions=self.regions, points=self.points,
                    refresh_steps=self.refresh_steps, last_refresh=self.last_refresh,
                    tables=self.tables, last_scale=self.last_scale,
                    last_packet_bytes=self.last_packet_bytes, last_entropy_bytes=self.last_entropy_bytes,
                    last_measure_step=self.last_measure_step)

    @classmethod
    def restore(cls, state):
        result = cls.__new__(cls)
        result.__dict__.update(state)
        result.previous_opened = state.get('previous_opened')
        result.last_measure_step = state.get('last_measure_step', -1)
        if result.tables is not None:
            result.tables = result.tables.detach().cpu()
        result._device_tables = {}
        return result
