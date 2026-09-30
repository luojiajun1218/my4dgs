"""Self-contained residual frame packets and differentiable rate estimates.

The residual entropy path uses lazy-loaded torchac with explicit 16-bit CDFs.
Its proxy uses the same discrete probabilities forward and continuous gradients.
No pickle or renderer is needed.
All bootstrap, quantization and layout information must be passed in metadata;
all supplied tensors (including masks, deletions and zero birth attributes) are
serialized. Decoder output is on CPU and requires no preceding frame packet.
"""

from functools import lru_cache
import json
import math
import os
from pathlib import Path
import struct
import tempfile

import numpy as np
import torch


PROTOCOL_VERSION = 1
PREFIX = struct.Struct("<8sI")
DTYPES = {
    "bool": np.dtype("?"),
    "uint8": np.dtype("u1"),
    "int8": np.dtype("i1"),
    "int16": np.dtype("<i2"),
    "int32": np.dtype("<i4"),
    "int64": np.dtype("<i8"),
    "float32": np.dtype("<f4"),
    "float64": np.dtype("<f8"),
}


def _laplace_mass(values, quant_steps, scales, *, validate=True):
    steps = torch.as_tensor(quant_steps, dtype=values.dtype, device=values.device)
    scales = torch.as_tensor(scales, dtype=values.dtype, device=values.device)
    if validate and (torch.any(steps <= 0) or torch.any(scales <= 0)):
        raise ValueError("quantization steps and Laplace scales must be positive")
    half = steps / 2
    absolute = values.abs()
    # Stable closed forms for bins wholly on one side and crossing zero.
    outside = (0.5 * torch.exp(-(absolute - half).clamp_min(0) / scales)
               * (-torch.expm1(-steps / scales)))
    inside = (-0.5 * torch.expm1(-(half - absolute).clamp_min(0) / scales)
              - 0.5 * torch.expm1(-(half + absolute) / scales))
    mass = torch.where(absolute >= half, outside, inside)
    return mass.clamp_min(torch.finfo(values.dtype).tiny)


ENTROPY_MAGIC = b'GHENT001'
ENTROPY_CHUNK = struct.Struct('<III')
UINT32 = struct.Struct('<I')


def _integer_cdf(probabilities):
    """Positive frequencies summing to 65536; last symbol absorbs remainder."""
    probabilities = probabilities.detach().to(device='cpu', dtype=torch.float64)
    count = probabilities.numel()
    if count > 32767:
        raise ValueError('alphabet too large for signed torchac symbols')
    frequencies = torch.floor(probabilities * (65536 - count)).to(torch.int64) + 1
    frequencies[-1] += 65536 - int(frequencies.sum())
    cdf = torch.cat((torch.zeros(1, dtype=torch.int64), frequencies.cumsum(0)))
    return cdf.to(torch.int16), frequencies.to(torch.float64) / 65536


def laplace_integer_cdf(scale, entropy_radius):
    """Get a cached CPU CDF for a fixed configuration scale."""
    return _cached_laplace_integer_cdf(float(scale), entropy_radius)


@lru_cache(maxsize=512)
def _cached_laplace_integer_cdf(scale, entropy_radius):
    """Finite alphabet [-radius,radius], followed by an escape symbol."""
    if type(entropy_radius) is not int or not 0 <= entropy_radius <= 16382:
        raise ValueError('entropy_radius must be in [0, 16382]')
    scale = torch.as_tensor(scale, dtype=torch.float64, device='cpu')
    if scale.numel() != 1 or not torch.isfinite(scale) or scale <= 0:
        raise ValueError('each Laplace scale must be finite and positive')
    centers = torch.arange(-entropy_radius, entropy_radius + 1, dtype=torch.float64)
    masses = _laplace_mass(centers, 1., scale)
    escape = torch.exp(torch.tensor(-(entropy_radius + .5)) / scale).reshape(1)
    return _integer_cdf(torch.cat((masses, escape)))


@lru_cache(maxsize=512)
def _mask_cdf(probability):
    if not math.isfinite(float(probability)) or not 0 <= float(probability) <= 1:
        raise ValueError('mask probability must be in [0,1]')
    return _integer_cdf(torch.tensor([1 - float(probability), float(probability)],
                                   dtype=torch.float64))


def _entropy_header(packet, metadata, entropy_models, entropy_radius, chunk_rows):
    if type(chunk_rows) is not int or chunk_rows <= 0:
        raise ValueError('chunk_rows must be positive')
    laplace_integer_cdf(1., entropy_radius)
    descriptors = []
    for name, tensor in sorted(packet.items()):
        dtype = str(tensor.dtype).removeprefix('torch.')
        if dtype not in DTYPES:
            raise ValueError('unsupported tensor dtype')
        desc = {'name': name, 'dtype': dtype, 'shape': list(tensor.shape)}
        if name in entropy_models:
            if tensor.is_floating_point() or tensor.dtype == torch.bool:
                raise ValueError('entropy symbols must be integer tensors')
            channels = tensor.shape[-1] if tensor.ndim > 1 else 1
            if channels == 0:
                # SH degree zero has legitimate (birth_count, 0) attributes.
                # Preserve its shape as an empty raw tensor without streams.
                descriptors.append(desc)
                continue
            rows = tensor.numel() // channels
            model = dict(entropy_models[name])
            scales = torch.as_tensor(model['scale'], dtype=torch.float64).reshape(-1)
            if scales.numel() == 1:
                scales = scales.expand(channels)
            if scales.numel() != channels:
                raise ValueError('Laplace scales must be scalar or one per channel')
            for scale in scales:
                laplace_integer_cdf(scale, entropy_radius)
            sparse = bool(model.get('sparse', False))
            probability = float(model['mask_probability']) if sparse else None
            if sparse:
                _mask_cdf(probability)
            desc.update({'rows': rows, 'channels': channels, 'model': {
                'scale': scales.tolist(), 'sparse': sparse,
                'mask_probability': probability}})
        descriptors.append(desc)
    header = {'version': PROTOCOL_VERSION, 'entropy_backend': 'torchac_int16',
              'radius': entropy_radius, 'chunk_rows': chunk_rows,
              'metadata': dict(metadata), 'tensors': descriptors}
    return json.dumps(header, separators=(',', ':'), sort_keys=True,
                      allow_nan=False).encode('utf-8')


def container_overhead_bits(packet, metadata, entropy_models, entropy_radius, chunk_rows):
    """Exact fixed/header cost; excludes raw tensors, masks, values and escapes."""
    header = _entropy_header(packet, metadata, entropy_models, entropy_radius, chunk_rows)
    chunk_headers = 0
    for desc in json.loads(header)['tensors']:
        if 'model' in desc:
            chunks = (desc['rows'] + chunk_rows - 1) // chunk_rows
            chunk_headers += chunks * (ENTROPY_CHUNK.size + UINT32.size * desc['channels'])
    return 8 * (PREFIX.size + len(header) + chunk_headers)


def encode_entropy_frame(path, packet, metadata, *, entropy_radius, chunk_rows,
                         entropy_models):
    """Encode same-model arithmetic masks/values with raw int32 escapes.

    Models map tensor names to {scale: scalar/list, sparse: bool,
    mask_probability: float}. Nonmodel tensors use their exact raw dtype.
    """
    import torchac
    header = _entropy_header(packet, metadata, entropy_models, entropy_radius, chunk_rows)
    payloads = []
    counts = {'mask_bytes': 0, 'value_bytes': 0, 'escape_bytes': 0,
              'raw_bytes': 0, 'chunk_header_bytes': 0}
    tensor_bytes = {}
    escape_symbol = 2 * entropy_radius + 1
    for desc in json.loads(header)['tensors']:
        tensor = packet[desc['name']].detach().cpu().contiguous()
        start_bytes = sum(counts.values())
        if 'model' not in desc:
            raw = tensor.numpy().astype(DTYPES[desc['dtype']], copy=False).tobytes()
            payloads.append(raw)
            counts['raw_bytes'] += len(raw)
        else:
            values = tensor.reshape(desc['rows'], desc['channels']).to(torch.int64)
            if values.numel() and (values.min() < -(2 ** 31) or values.max() >= 2 ** 31):
                raise ValueError('entropy integer symbols exceed int32 range')
            model = desc['model']
            cdfs = [laplace_integer_cdf(scale, entropy_radius)[0] for scale in model['scale']]
            for start in range(0, desc['rows'], chunk_rows):
                chunk = values[start:start + chunk_rows]
                mask = (chunk != 0).any(-1) if model['sparse'] else torch.ones(chunk.shape[0], dtype=torch.bool)
                if model['sparse']:
                    cdf = _mask_cdf(model['mask_probability'])[0]
                    mask_bytes = torchac.encode_int16_normalized_cdf(
                        cdf.expand(mask.numel(), -1).contiguous(), mask.to(torch.int16))
                else:
                    mask_bytes = b''
                active = chunk[mask]
                streams, escapes = [], []
                for channel, cdf in enumerate(cdfs):
                    column = active[:, channel]
                    outside = column.abs() > entropy_radius
                    symbols = torch.where(outside, escape_symbol, column + entropy_radius).to(torch.int16)
                    encoded = (torchac.encode_int16_normalized_cdf(
                        cdf.expand(symbols.numel(), -1).contiguous(), symbols)
                               if symbols.numel() else b'')
                    streams.append(encoded)
                    escapes.append(column[outside].numpy().astype('<i4').tobytes())
                escape_bytes = b''.join(escapes)
                prefix = ENTROPY_CHUNK.pack(len(mask_bytes), active.shape[0], len(escape_bytes) // 4)
                lengths = b''.join(UINT32.pack(len(encoded)) for encoded in streams)
                payloads.extend([prefix, lengths, mask_bytes, *streams, escape_bytes])
                counts['chunk_header_bytes'] += len(prefix) + len(lengths)
                counts['mask_bytes'] += len(mask_bytes)
                counts['value_bytes'] += sum(map(len, streams))
                counts['escape_bytes'] += len(escape_bytes)
        tensor_bytes[desc['name']] = sum(counts.values()) - start_bytes
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = stream.name
            stream.write(PREFIX.pack(ENTROPY_MAGIC, len(header)))
            stream.write(header)
            for payload in payloads:
                stream.write(payload)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            os.unlink(temporary)
    header_bytes = PREFIX.size + len(header)
    total = header_bytes + sum(counts.values())
    return {'total_bytes': total, 'total_bits': total * 8, 'header_bytes': header_bytes,
            'tensor_bytes': tensor_bytes, 'codec': 'torchac_int16', **counts}


def decode_entropy_frame(path):
    """Decode dense original tensors; probability models live in the packet."""
    import torchac
    def read_exact(stream, size):
        data = stream.read(size)
        if len(data) != size:
            raise ValueError('truncated entropy packet')
        return data
    with open(path, 'rb') as stream:
        magic, length = PREFIX.unpack(read_exact(stream, PREFIX.size))
        if magic != ENTROPY_MAGIC:
            raise ValueError('invalid entropy magic')
        header = json.loads(read_exact(stream, length))
        if header['version'] != PROTOCOL_VERSION or header['entropy_backend'] != 'torchac_int16':
            raise ValueError('unsupported entropy protocol')
        tensors = {}
        radius, chunk_rows = header['radius'], header['chunk_rows']
        for desc in header['tensors']:
            dtype = DTYPES[desc['dtype']]
            if 'model' not in desc:
                raw = read_exact(stream, math.prod(desc['shape']) * dtype.itemsize)
                array = np.frombuffer(raw, dtype=dtype).copy().reshape(desc['shape'])
            else:
                model = desc['model']
                rows, channels = desc['rows'], desc['channels']
                array = np.zeros((rows, channels), dtype=dtype)
                cdfs = [laplace_integer_cdf(scale, radius)[0] for scale in model['scale']]
                for start in range(0, rows, chunk_rows):
                    n = min(chunk_rows, rows - start)
                    mask_length, active_count, escape_count = ENTROPY_CHUNK.unpack(read_exact(stream, ENTROPY_CHUNK.size))
                    lengths = [UINT32.unpack(read_exact(stream, UINT32.size))[0] for _ in range(channels)]
                    mask_bytes = read_exact(stream, mask_length)
                    if model['sparse']:
                        cdf = _mask_cdf(model['mask_probability'])[0]
                        mask = torchac.decode_int16_normalized_cdf(cdf.expand(n, -1).contiguous(), mask_bytes).bool()
                    else:
                        if mask_length:
                            raise ValueError('unexpected dense mask stream')
                        mask = torch.ones(n, dtype=torch.bool)
                    if int(mask.sum()) != active_count:
                        raise ValueError('mask activity count mismatch')
                    columns = []
                    for cdf, encoded_length in zip(cdfs, lengths):
                        encoded = read_exact(stream, encoded_length)
                        if active_count:
                            symbols = torchac.decode_int16_normalized_cdf(cdf.expand(active_count, -1).contiguous(), encoded).to(torch.int64)
                        else:
                            if encoded_length:
                                raise ValueError('unexpected empty scalar stream')
                            symbols = torch.empty(0, dtype=torch.int64)
                        columns.append(symbols)
                    escapes = np.frombuffer(read_exact(stream, escape_count * 4), dtype='<i4')
                    position = 0
                    active = np.empty((active_count, channels), dtype=dtype)
                    for channel, symbols in enumerate(columns):
                        outside = symbols == 2 * radius + 1
                        count = int(outside.sum())
                        column = symbols - radius
                        column[outside] = torch.from_numpy(escapes[position:position + count].copy()).to(torch.int64)
                        position += count
                        active[:, channel] = column.numpy()
                    if position != escape_count:
                        raise ValueError('escape count mismatch')
                    array[start:start + n][mask.numpy()] = active
                array = array.reshape(desc['shape'])
            if desc['name'] in tensors:
                raise ValueError('duplicate tensor name')
            tensors[desc['name']] = torch.from_numpy(array.copy())
        if stream.read(1):
            raise ValueError('unexpected trailing entropy bytes')
    return {'tensors': tensors, 'metadata': header['metadata']}


@lru_cache(maxsize=512)
def _device_laplace_frequencies(scale, radius, device, dtype):
    return laplace_integer_cdf(scale, radius)[1].to(device=device, dtype=dtype)


def expected_torchac_bits(symbols_ST, scale, *, entropy_radius,
                          mask_probability=None, support_temperature=None, sparse=False,
                          reduction='sum'):
    """Same 16-bit frequencies forward; continuous Laplace/support STE gradient.

    Scalars in active tuples use unconditional channel distributions, exactly
    as the arithmetic coder. Escape symbols additionally retain 32 raw bits.
    Headers, arithmetic termination and unmodeled tensors are charged separately.
    """
    if reduction not in ('sum', 'none'):
        raise ValueError("reduction must be 'sum' or 'none'")
    if symbols_ST.numel() == 0:
        if reduction == 'sum':
            return symbols_ST.sum() * 0
        rows = math.prod(symbols_ST.shape[:-1]) if symbols_ST.ndim > 1 else symbols_ST.numel()
        return symbols_ST.new_zeros(rows) + symbols_ST.sum() * 0
    values = symbols_ST.reshape(-1, symbols_ST.shape[-1] if symbols_ST.ndim > 1 else 1)
    # Model scales are fixed config values; transfer a tensor input only once,
    # never synchronize GPU individually for every channel.
    if isinstance(scale, torch.Tensor):
        scale_list = scale.detach().cpu().reshape(-1).tolist()
    elif isinstance(scale, (float, int)):
        scale_list = [float(scale)]
    else:
        scale_list = [float(item) for item in scale]
    if len(scale_list) == 1:
        scale_list *= values.shape[1]
    if len(scale_list) != values.shape[1]:
        raise ValueError('scale count does not match channels')
    hard = values.detach().round()
    scalar_bits = []
    for channel, channel_scale in enumerate(scale_list):
        frequencies = _device_laplace_frequencies(
            channel_scale, entropy_radius, str(values.device), values.dtype)
        outside = hard[:, channel].abs() > entropy_radius
        indices = torch.where(outside, 2 * entropy_radius + 1,
                              hard[:, channel] + entropy_radius).to(torch.int64)
        exact = -torch.log2(frequencies[indices]) + outside.to(values.dtype) * 32
        # Cached CPU construction already checked this fixed scale; do not
        # perform scalar validation on GPU during each rate-loss call.
        in_range_proxy = -torch.log2(_laplace_mass(
            values[:, channel], 1., channel_scale, validate=False))
        escape_proxy = (entropy_radius + .5) / channel_scale / math.log(2) + 32
        proxy = torch.where(outside, escape_proxy, in_range_proxy)
        scalar_bits.append(exact + (proxy - proxy.detach()))
    value_cost = torch.stack(scalar_bits, dim=-1).sum(-1)
    if not sparse:
        return value_cost.sum() if reduction == 'sum' else value_cost
    if mask_probability is None or support_temperature is None or support_temperature <= 0:
        raise ValueError('sparse rate requires mask_probability and positive support_temperature')
    mask = (hard != 0).any(-1).to(values.dtype)
    support = 1 - torch.exp(-values.abs().sum(-1) / support_temperature)
    gate = mask + (support - support.detach())
    _, frequencies = _mask_cdf(mask_probability)
    costs = -torch.log2(frequencies.to(device=values.device, dtype=values.dtype))
    row_cost = gate * value_cost + gate * costs[1] + (1 - gate) * costs[0]
    return row_cost.sum() if reduction == 'sum' else row_cost
