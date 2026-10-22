"""Read-only replay of DC updates written before shared-color removal.

The active trainer has no color-update parameters, optimizer, or encoder.
Keep this decoder so completed historical experiments remain reproducible.
"""

import hashlib
import struct

import torch

from .legacy_opacity_codec import decode_opacity_delta


MAGIC = b'SDC1'
REGIONAL_MAGIC = b'SDR1'


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
        # Preserve signed zeros and every reference bit for zero increments.
        return torch.where(delta.ne(0), reference + delta, reference)
    if packet[:4] != MAGIC:
        raise ValueError('Unknown DC feature update stream')
    decoded = decode_opacity_delta(packet[4:], reference.reshape(-1, 1), frame_index=frame_index)
    if decoded.mode != 'logit':
        raise ValueError('DC updates require additive linear symbols')
    return decoded.logits.reshape_as(reference)
