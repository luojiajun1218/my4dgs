"""Quantized color reuse on a sparse set of existing Gaussians."""

import struct
import zlib
from pathlib import Path

import numpy as np
import torch
from torch import nn

MAGIC = b'SCD1'
HEADER = struct.Struct('<4sIIdII')


class SparseColorUpdate(nn.Module):
    def __init__(self, base_dc, rows, step, lr, support_temperature):
        super().__init__()
        rows = torch.as_tensor(rows, device=base_dc.device, dtype=torch.long)
        if (base_dc.ndim != 3 or base_dc.shape[1:] != (1, 3) or rows.ndim != 1 or
                len(torch.unique(rows)) != len(rows) or bool(((rows < 0) | (rows >= len(base_dc))).any())):
            raise ValueError('Sparse color update requires unique valid rows and DC [N,1,3]')
        self.base_count = len(base_dc)
        self.step, self.lr, self.support_temperature = float(step), float(lr), float(support_temperature)
        if min(self.step, self.lr, self.support_temperature) <= 0:
            raise ValueError('Sparse color settings must be positive')
        self.register_buffer('rows', rows)
        self.register_buffer('reference', base_dc[rows].detach().clone())
        self.delta = nn.Parameter(base_dc.new_zeros(len(rows), 3))
        self.optimizer = torch.optim.Adam([self.delta], lr=self.lr, eps=1e-15)

    def symbols(self):
        return torch.round(self.delta / self.step).clamp(-32768, 32767)

    def forward(self, base_dc):
        symbols = self.symbols()
        decoded = symbols * self.step + (self.delta - self.delta.detach())
        return base_dc.index_copy(0, self.rows, self.reference + decoded[:, None, :])

    def optimizer_step(self):
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

    def encode(self):
        symbols = self.symbols().detach()
        active = symbols.ne(0).any(-1)
        if not bool(active.any()):
            return b''
        rows = self.rows[active]
        order = torch.argsort(rows, stable=True)
        ids = rows[order].cpu().numpy().astype('<u4')
        codes = symbols[active][order].cpu().numpy().astype('<i2')
        body = ids.tobytes() + codes.tobytes()
        return HEADER.pack(MAGIC, self.base_count, len(ids), self.step, len(body),
                           zlib.crc32(body)) + zlib.compress(body, level=3)

    def storage_bytes(self, hard=False):
        if hard:
            return self.delta.new_tensor(float(len(self.encode())))
        scaled = self.delta / self.step
        hard_support = self.symbols().ne(0).any(-1).to(self.delta)
        smooth_zero = torch.sigmoid((.5 - scaled.abs()) / self.support_temperature).prod(-1)
        support = hard_support + ((1 - smooth_zero) - (1 - smooth_zero).detach())
        present = 1 - (1 - support).prod()
        return support.sum() * 10 + present * HEADER.size

    def export(self, path):
        packet = self.encode()
        if not packet:
            return None
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(packet)
        return dict(path=str(path), archive_bytes=len(packet),
                    rows=int(self.symbols().ne(0).any(-1).sum()), format='sparse_dc_zlib_v1')

    def capture(self):
        return dict(base_count=self.base_count, step=self.step, lr=self.lr,
                    support_temperature=self.support_temperature, state=self.state_dict(),
                    optimizer=self.optimizer.state_dict())

    @classmethod
    def restore(cls, packet, base_dc):
        obj = cls(base_dc, packet['state']['rows'], packet['step'], packet['lr'],
                  packet['support_temperature'])
        if obj.base_count != packet['base_count']:
            raise ValueError('Sparse color checkpoint base count mismatch')
        obj.load_state_dict(packet['state'])
        obj.optimizer.load_state_dict(packet['optimizer'])
        return obj


def apply_sparse_color_packet(path_or_packet, raw):
    packet = (bytes(path_or_packet) if isinstance(path_or_packet, (bytes, bytearray))
              else Path(path_or_packet).read_bytes())
    if len(packet) < HEADER.size:
        raise ValueError('Truncated sparse color header')
    magic, count, rows, step, length, checksum = HEADER.unpack_from(packet)
    if magic != MAGIC or count != len(raw['xyz']) or rows > count or length != rows * 10 or not np.isfinite(step) or step <= 0:
        raise ValueError('Invalid sparse color header')
    body = zlib.decompress(packet[HEADER.size:])
    if len(body) != length or zlib.crc32(body) != checksum:
        raise ValueError('Sparse color payload failed validation')
    ids = np.frombuffer(body[:rows * 4], dtype='<u4').astype(np.int64)
    if np.any(ids >= count) or (len(ids) > 1 and np.any(ids[1:] <= ids[:-1])):
        raise ValueError('Invalid sparse color row IDs')
    codes = np.frombuffer(body[rows * 4:], dtype='<i2').reshape(rows, 3).copy()
    device, dtype = raw['sh_dc'].device, raw['sh_dc'].dtype
    indices = torch.as_tensor(ids.copy(), device=device, dtype=torch.long)
    delta = torch.as_tensor(codes, device=device, dtype=dtype) * step
    color = raw['sh_dc'].index_copy(0, indices, raw['sh_dc'][indices] + delta[:, None, :])
    return dict(raw, sh_dc=color)
