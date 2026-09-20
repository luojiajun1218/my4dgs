"""Quantized motion, bin-integrated Laplace rate, and a real torchac codec.

Independent implementation. See docs/adaptive_rd_worklog.md for conceptual
references (QUEEN/HAC/RD4DGS). No trained network or hidden decoder state required.
"""
import io
import json
import math

import numpy as np
import torch


def quantize_ste(value, step):
    rounded = torch.round(value / step) * step
    return value + (rounded - value).detach()


def rotation_vector_to_quaternion(vector):
    """[N,3] axis-angle increments -> scalar-first [N,4], safe at zero."""
    angle = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
    xyz = vector * (0.5 * torch.sinc(angle / (2 * math.pi)))
    return torch.cat([torch.cos(angle / 2), xyz], dim=-1)


def laplace_bits(symbols, scale):
    """Discrete bin mass, not differential density. Scale is in symbol units."""
    scale = scale.clamp_min(0.1)
    magnitude = symbols.abs()
    # Reflect positive bins into the stable negative tail of the symmetric CDF.
    upper = (0.5 - magnitude) / scale
    lower = (-0.5 - magnitude) / scale
    cdf_upper = torch.where(upper <= 0, 0.5 * upper.clamp_max(0).exp(),
                            1 - 0.5 * (-upper.clamp_min(0)).exp())
    cdf_lower = 0.5 * lower.exp()
    probability = cdf_upper - cdf_lower
    # Straight-through floor retains the useful direction in extreme tails.
    bounded = probability + (probability.clamp_min(1e-9) - probability).detach()
    return -torch.log2(bounded)


def fit_laplace_scales(symbols, groups, num_groups):
    """Closed-form zero-mean Laplace fit; decoder receives exact CDF tables."""
    totals = symbols.new_zeros(num_groups, symbols.shape[1])
    totals.index_add_(0, groups, symbols.detach().abs())
    counts = torch.bincount(groups, minlength=num_groups).clamp_min(1)
    return (totals / counts[:, None]).clamp_min(0.1)


def _integer_cdf(low, high, scale):
    # Positive count for every symbol, stored as exact uint16 bit patterns.
    bins = high - low + 1
    if bins > 32767:
        raise ValueError('Symbol range exceeds torchac int16 alphabet')
    edges = torch.arange(low, high + 2, dtype=torch.float64) - 0.5
    x = edges / max(float(scale), 0.1)
    cdf = torch.where(x <= 0, 0.5 * x.clamp_max(0).exp(),
                      1 - 0.5 * (-x.clamp_min(0)).exp())
    cdf[0], cdf[-1] = 0, 1
    integer = torch.round(cdf * (65536 - bins)).long() + torch.arange(bins + 1)
    return integer.to(torch.int16)


def encode_coefficients(symbols, groups, scales, chunk_size=1024):
    """Encode [M,6] integers, transmit exact CDFs; lossless raw fallback for outliers."""
    import torchac

    symbols = np.asarray(symbols, dtype=np.int32)
    arrays, records = {}, []
    for group in np.unique(groups):
        rows = np.flatnonzero(groups == group)
        for channel in range(symbols.shape[1]):
            values = symbols[rows, channel]
            lo, hi = int(values.min()), int(values.max())
            key = f'g{group}_c{channel}'
            record = dict(group=int(group), channel=channel, low=lo, high=hi, key=key)
            if hi - lo >= 32767:
                record['raw'] = True
                arrays[key] = values
            elif lo == hi:
                record['constant'] = lo
            else:
                cdf = _integer_cdf(lo, hi, scales[group, channel])
                arrays[key + '_cdf'] = cdf.numpy()
                record['chunks'] = []
                for start in range(0, len(values), chunk_size):
                    block = torch.from_numpy((values[start:start + chunk_size] - lo).astype(np.int16))
                    table = cdf[None].expand(len(block), -1).contiguous()
                    stream = torchac.encode_int16_normalized_cdf(table, block)
                    chunk_key = key + f'_{start}'
                    arrays[chunk_key] = np.frombuffer(stream, dtype=np.uint8)
                    record['chunks'].append([chunk_key, len(block)])
            records.append(record)
    arrays['codec_records'] = np.array(json.dumps(records))
    arrays['coefficient_groups'] = np.asarray(groups, dtype=np.int16)
    return arrays


def decode_coefficients(archive):
    import torchac

    groups = archive['coefficient_groups']
    values = np.zeros((len(groups), 6), dtype=np.int32)
    for record in json.loads(str(archive['codec_records'])):
        rows = np.flatnonzero(groups == record['group'])
        key = record['key']
        if record.get('raw'):
            decoded = archive[key]
        elif 'constant' in record:
            decoded = np.full(len(rows), record['constant'], dtype=np.int32)
        else:
            cdf = torch.from_numpy(archive[key + '_cdf'].copy())
            parts = []
            for chunk_key, count in record['chunks']:
                table = cdf[None].expand(count, -1).contiguous()
                result = torchac.decode_int16_normalized_cdf(table, archive[chunk_key].tobytes())
                parts.append(result.numpy().astype(np.int32) + record['low'])
            decoded = np.concatenate(parts)
        values[rows, record['channel']] = decoded
    return values


def structure_arrays(root_index, split_nodes, path_index, path_weight):
    roots = root_index.detach().cpu().numpy()
    dtype = np.uint16 if roots.max() < 65536 else np.uint32
    arrays = dict(root_index=roots.astype(dtype),
                  split_nodes=split_nodes.detach().cpu().numpy().astype(np.int32))
    for depth in range(path_index.shape[1]):
        present = path_index[:, depth] >= 0
        choices = (path_weight[present, depth] < 0).detach().cpu().numpy()
        arrays[f'branch_{depth}'] = np.packbits(choices, bitorder='little')
    return arrays


def compressed_structure_bits(arrays):
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    return len(stream.getvalue()) * 8


def decode_motion(path, device='cpu'):
    """Self-contained motion decode; GS ordering is the only external convention."""
    with np.load(path, allow_pickle=False) as archive:
        roots = archive['root_index'].astype(np.int64)
        splits = archive['split_nodes'].astype(np.int64)
        root_count, max_depth = map(int, archive['tree_shape'])
        coefficients = decode_coefficients(archive).astype(np.float32) * archive['quant_step']
        child = np.full((root_count + 2 * len(splits), 2), -1, dtype=np.int64)
        split_coefficient = np.full(len(child), -1, dtype=np.int64)
        child[splits] = root_count + np.arange(2 * len(splits)).reshape(-1, 2)
        split_coefficient[splits] = root_count + np.arange(len(splits))
        leaf = roots.copy()
        field = coefficients[roots].copy()
        for depth in range(max_depth):
            present = child[leaf, 0] >= 0
            parents = leaf[present]
            if not present.any():
                break
            right = np.unpackbits(archive[f'branch_{depth}'], bitorder='little')[:len(parents)]
            next_leaf = child[parents, right]
            count = np.bincount(next_leaf, minlength=len(child))
            nl, nr = count[child[parents, 0]], count[child[parents, 1]]
            weight = np.where(right == 0, nr, -nl).astype(np.float32) / (nl + nr)
            field[present] += coefficients[split_coefficient[parents]] * weight[:, None]
            leaf[present] = next_leaf
    tensor = torch.as_tensor(field, device=device)
    return tensor[:, :3].contiguous(), rotation_vector_to_quaternion(tensor[:, 3:])
