"""Quantized opacity fitting and deterministic expiration for historical births."""

from pathlib import Path
import struct
import zlib

import numpy as np
import torch
from torch import nn


MAGIC = b'OPA1'
HEADER = struct.Struct('<4sIIdII')


class BirthOpacityUpdate(nn.Module):
    def __init__(self, base, rows, step, lr, support_temperature):
        super().__init__()
        rows = torch.as_tensor(rows, device=base.device, dtype=torch.long)
        if base.shape != (len(base), 1) or rows.ndim != 1 or len(rows.unique()) != len(rows) or bool(((rows < 0) | (rows >= len(base))).any()):
            raise ValueError('Birth opacity update requires unique valid rows and logits [N,1]')
        self.base_count, self.step, self.lr = len(base), float(step), float(lr)
        self.support_temperature = float(support_temperature)
        if not np.isfinite([self.step, self.lr, self.support_temperature]).all() or min(self.step, self.lr, self.support_temperature) <= 0:
            raise ValueError('Birth opacity settings must be finite and positive')
        self.register_buffer('rows', rows)
        self.register_buffer('reference', base[rows].detach().clone())
        self.delta = nn.Parameter(base.new_zeros(len(rows), 1))
        self.optimizer = torch.optim.Adam([self.delta], lr=self.lr, eps=1e-15)

    def symbols(self):
        return torch.round(self.delta / self.step).clamp(-32768, 32767)

    def values(self):
        decoded = self.symbols() * self.step + (self.delta - self.delta.detach())
        return self.reference + decoded

    def forward(self, base):
        return base.index_copy(0, self.rows, self.values())

    def optimizer_step(self):
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

    def storage_bytes(self, hard=False):
        if hard:
            return self.delta.new_tensor(float(len(self.encode())))
        hard_support = self.symbols().ne(0).flatten().to(self.delta)
        smooth = torch.sigmoid((self.delta.flatten().abs() / self.step - .5) / self.support_temperature)
        support = hard_support + smooth - smooth.detach()
        return support.sum() * 6 + (1 - (1 - support).prod()) * HEADER.size

    def encode(self):
        code = self.symbols().detach().flatten()
        active = code != 0
        if not bool(active.any()):
            return b''
        rows = self.rows[active]
        order = torch.argsort(rows, stable=True)
        ids = rows[order].cpu().numpy().astype('<u4')
        symbols = code[active][order].cpu().numpy().astype('<i2')
        body = ids.tobytes() + symbols.tobytes()
        return HEADER.pack(MAGIC, self.base_count, len(ids), self.step, len(body),
                           zlib.crc32(body)) + zlib.compress(body, 3)

    def export(self, path):
        packet = self.encode()
        if not packet:
            return None
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(packet)
        return dict(path=str(path), archive_bytes=len(packet),
                    rows=int(self.symbols().ne(0).sum()), format='birth_opacity_int16_zlib_v1')

    def capture(self):
        return dict(base_count=self.base_count, step=self.step, lr=self.lr,
                    support_temperature=self.support_temperature,
                    state=self.state_dict(), optimizer=self.optimizer.state_dict())

    @classmethod
    def restore(cls, packet, base):
        obj = cls(base, packet['state']['rows'], packet['step'], packet['lr'], packet['support_temperature'])
        if obj.base_count != packet['base_count']:
            raise ValueError('Birth opacity checkpoint count mismatch')
        obj.load_state_dict(packet['state'])
        obj.optimizer.load_state_dict(packet['optimizer'])
        return obj


def apply_opacity_packet(path, raw):
    packet = bytes(path) if isinstance(path, (bytes, bytearray)) else Path(path).read_bytes()
    if len(packet) < HEADER.size:
        raise ValueError('Truncated birth opacity stream')
    magic, base_count, count, step, length, checksum = HEADER.unpack_from(packet)
    if (magic != MAGIC or base_count != len(raw['xyz']) or count > base_count or
            length != count * 6 or length > 256 * 1024 * 1024 or not np.isfinite(step) or step <= 0):
        raise ValueError('Invalid birth opacity header')
    decoder = zlib.decompressobj()
    body = decoder.decompress(packet[HEADER.size:], length + 1)
    if len(body) != length or not decoder.eof or decoder.unused_data or decoder.unconsumed_tail or zlib.crc32(body) != checksum:
        raise ValueError('Birth opacity payload failed validation')
    ids = np.frombuffer(body[:count * 4], dtype='<u4').astype(np.int64)
    if np.any(ids >= base_count) or (count > 1 and np.any(ids[1:] <= ids[:-1])):
        raise ValueError('Invalid birth opacity row IDs')
    codes = np.frombuffer(body[count * 4:], dtype='<i2').copy()
    rows = torch.as_tensor(ids.copy(), device=raw['opacity'].device, dtype=torch.long)
    delta = torch.as_tensor(codes, device=rows.device, dtype=raw['opacity'].dtype)[:, None] * step
    return dict(raw, opacity=raw['opacity'].index_copy(0, rows, raw['opacity'][rows] + delta))


def expiration_deletions(opacity, previous_birth_count, previous_changed, birth_frames, threshold):
    """Remove expired births first, then compensate remaining birth capacity.

    Expired historical births can override pose-KEEP. Below-threshold shared
    points can supply compensation capacity; they are not forcibly expired.
    """
    total = len(opacity)
    origins = torch.as_tensor(birth_frames, device=opacity.device, dtype=torch.long)
    old_count = total - previous_birth_count
    if (origins.shape != (total,) or previous_birth_count < 0 or old_count < 1 or
            previous_changed.shape != (old_count,) or not 0 < threshold < 1):
        raise ValueError('Birth expiration inputs disagree')
    inactive = torch.sigmoid(opacity.detach().flatten()) < threshold
    expired = (origins > 0) & inactive
    protected = torch.zeros(total, device=opacity.device, dtype=torch.bool)
    protected[:old_count] = ~previous_changed.to(device=opacity.device, dtype=torch.bool)
    expired_rows = torch.where(expired)[0]
    target = max(previous_birth_count, len(expired_rows))
    eligible = ~protected | ((origins <= 0) & inactive)
    candidates = torch.where(eligible & ~expired)[0]
    remaining = target - len(expired_rows)
    if len(candidates) < remaining:
        raise ValueError('Insufficient eligible rows for birth compensation')
    chosen = candidates[torch.argsort(opacity.detach().flatten()[candidates], stable=True)[:remaining]]
    deleted = torch.cat((expired_rows, chosen))
    return deleted, dict(expired_births=len(expired_rows), compensation_deleted=len(chosen),
        protected_keep_old_count=int(protected.sum()), eligible_count=int((eligible | expired).sum()),
        requested_count=target, deleted_count=len(deleted), deleted_keep_count=int(protected[deleted].sum()),
        capacity_deleted_invisible_first_frame=int(((origins[chosen] <= 0) & protected[chosen]).sum()),
        semantics='Expire learned low-opacity births first; compensate any remaining prior-birth capacity. '
                  'Expired births may override pose-KEEP; below-threshold shared points can supply '
                  'compensation capacity. Visible shared first-frame KEEP rows stay protected.')
