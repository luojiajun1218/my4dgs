"""顺序解码帧码流，并维护供渲染器使用的播放状态。

GPUStream 从共享首帧开始读取各帧更新，维护 GPU 上的 GS 属性并提供渲染视图；
命令行支持 CPU 参考解码、GPU 状态核对及导出指定帧。码流格式和基础解码逻辑
由 pipeline.model.frame_stream_format 和 frame_stream_decode 负责。
"""


import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline.model.frame_stream_decode import decode_frame, replay
from pipeline.model.frame_stream_format import RAW_FIELDS, reference_digest
from pipeline.model.frame_stream_format import (DecodedState, _validate_state)


def physical_covariance(scale, rotation, modifier=1.):
    """Compute the covariance needed to render a decoded Gaussian state."""
    w, x, y, z = F.normalize(rotation, dim=-1).unbind(-1)
    matrix = torch.stack((1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
        2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
        2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)), -1).reshape(-1, 3, 3)
    basis = matrix * (scale * modifier)[:, None, :]
    covariance = basis @ basis.transpose(1, 2)
    return covariance[:, (0, 0, 0, 1, 1, 2), (0, 1, 2, 1, 2, 2)]


class StreamView:
    def __init__(self, values, sh_degree):
        self.values=values
        self.active_sh_degree=sh_degree
        self.max_sh_degree=sh_degree

    @property
    def get_xyz(self): return self.values['xyz']
    @property
    def get_rotation(self): return self.values['rotation']
    @property
    def get_scaling(self): return self.values['scaling']
    @property
    def get_opacity(self): return self.values['opacity']
    @property
    def get_features(self): return self.values['features']
    def get_covariance(self, scaling_modifier=1.):
        return physical_covariance(self.get_scaling,self.get_rotation,scaling_modifier)


class GPUStream:
    """Keep immutable attributes in stable-ID banks, update only changed rows."""
    @torch.no_grad()
    def __init__(self, reference_state, raw, *, capacity, device='cuda'):
        _validate_state(reference_state)
        if reference_state.frame_index != 0 or reference_state.spatial_centers is None:
            raise ValueError('GPU stream needs the spatial frame0 reference')
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is required for incremental GPU streaming')
        if capacity < len(reference_state.xyz):
            raise ValueError('Insufficient stable-ID bank capacity')
        self.reference=reference_state
        self._portable_state = reference_state
        self._portable_raw = {name: value.detach().cpu().clone() for name, value in raw.items()}
        self.depth=reference_state.spatial_paths.shape[1]
        self.rest=raw['sh_rest'].shape[1]
        side=math.isqrt(1+self.rest)
        self.sh_degree=side-1
        if side*side != 1+self.rest or not 0 <= self.sh_degree <= 3:
            raise ValueError('GPU stream requires a complete SH0-SH3 coefficient set')
        count=len(reference_state.xyz)
        shapes=dict(xyz=(count,3),rotation=(count,4),logscale=(count,3),opacity=(count,1),
                    sh_dc=(count,1,3),sh_rest=(count,self.rest,3))
        if any(raw[name].dtype != torch.float32 or tuple(raw[name].shape) != shape or
               not bool(torch.isfinite(raw[name]).all()) for name,shape in shapes.items()):
            raise ValueError('Reference raw attributes need aligned finite float32 fields')
        self.device=torch.device(device)
        self.capacity=int(capacity)
        self.ids=np.arange(len(reference_state.xyz),dtype=np.int64)
        self._dense_ids = True
        self.next_uid=len(self.ids)
        self.frame=0
        self.expected_digest=reference_state.digest()
        self.xyz_bank=torch.empty((capacity,3),device=self.device)
        self.xyz_bank[:len(self.ids)].copy_(raw['xyz'])
        self.attribute_bank={name:torch.empty(shape,device=self.device) for name,shape in
            dict(rotation=(capacity,4),scaling=(capacity,3),opacity=(capacity,1),
                 features=(capacity,1+self.rest,3)).items()}
        self.raw_attribute_bank = {
            name: torch.empty((capacity, *raw[name].shape[1:]), dtype=raw[name].dtype,
                              device=self.device)
            for name in ('rotation', 'logscale', 'opacity', 'sh_dc', 'sh_rest')
        }
        for name, bank in self.raw_attribute_bank.items():
            bank[:count].copy_(raw[name])
        self.slots=[]
        for _ in range(2):
            self.slots.append(dict(ids=torch.empty(capacity,device=self.device,dtype=torch.long),
                xyz=torch.empty((capacity,3),device=self.device),
                owner=torch.empty(capacity,device=self.device,dtype=torch.long),
                paths=torch.empty((capacity,self.depth),device=self.device,dtype=torch.long),
                **{name:torch.empty_like(value) for name,value in self.attribute_bank.items()}))
        self.slot=0
        self.slots[0]['ids'][:len(self.ids)].copy_(torch.from_numpy(self.ids))
        self.slots[0]['xyz'][:len(self.ids)].copy_(raw['xyz'])
        self.slots[0]['owner'][:len(self.ids)].copy_(reference_state.owner)
        self.slots[0]['paths'][:len(self.ids)].copy_(reference_state.spatial_paths)
        self.last_open=reference_state.spatial_open.clone()
        self.last_sources=np.empty(0,np.int64)
        self.last_survivors=np.arange(len(self.ids),dtype=np.int64)
        self.attribute_bank['rotation'][:count].copy_(F.normalize(self.raw_attribute_bank['rotation'][:count], dim=-1))
        self.attribute_bank['scaling'][:count].copy_(self.raw_attribute_bank['logscale'][:count].exp())
        self.attribute_bank['opacity'][:count].copy_(self.raw_attribute_bank['opacity'][:count].sigmoid())
        self.attribute_bank['features'][:count].copy_(torch.cat(
            (self.raw_attribute_bank['sh_dc'][:count], self.raw_attribute_bank['sh_rest'][:count]), dim=1))
        self.view=self._view()

    def _upload(self, array):
        tensor=torch.from_numpy(np.ascontiguousarray(array))
        # PyTorch's pinned allocator records asynchronous copy lifetimes.
        return tensor.pin_memory().to(self.device,non_blocking=True)


    @torch.no_grad()
    def advance_frame(self, directory, frame):
        """Decode a complete frame and update the GPU renderer's GS buffers.

        GPU raw state stays resident and receives motion, rotation, opacity and
        appearance updates. CPU decoding maintains an independent canonical
        reference for the current packet hashes; it is not uploaded as the GPU
        state for the supported full-attribute path.
        """
        if frame != self.frame + 1:
            raise ValueError('Frames must be played in order')
        directory = Path(directory)
        current_ids = self.slots[self.slot]['ids'][:len(self.ids)]
        # With no deletion, stable IDs are 0..N-1 and the banks themselves
        # are already in frame order. Slicing avoids gathering every field.
        if self._dense_ids:
            gpu_reference = {name: bank[:len(self.ids)]
                             for name, bank in self.raw_attribute_bank.items()}
            gpu_reference['xyz'] = self.xyz_bank[:len(self.ids)]
        else:
            gpu_reference = {name: bank.index_select(0, current_ids)
                             for name, bank in self.raw_attribute_bank.items()}
            gpu_reference['xyz'] = self.xyz_bank.index_select(0, current_ids)
        gpu_capture = {}
        manifest = json.loads((directory / f'frame_{frame:03d}.json').read_text())
        if manifest.get('frame') != frame:
            raise ValueError('Frame manifest does not match the requested frame')
        decoded, raw = decode_frame(directory, frame, self._portable_state,
                                    self._portable_raw, attribute_device=self.device,
                                    gpu_reference=gpu_reference, gpu_capture=gpu_capture,
                                    verified_prior_digest=self.expected_digest,
                                    manifest=manifest)
        decoded_digest = decoded.digest()
        if (decoded_digest.hex() != manifest['decoded_state_sha256'] or
                reference_digest(raw) != manifest['raw_state_sha256']):
            raise ValueError('Decoded frame differs from its manifest')
        rows = decoded.survivor_rows.numpy()
        birth_count = len(decoded.xyz) - len(rows)
        if self.next_uid + birth_count > self.capacity:
            raise ValueError('Stable-ID bank capacity exceeded')
        all_survive = len(rows) == len(self.ids)
        dense_ids = self._dense_ids and all_survive
        if all_survive and not birth_count:
            ids = self.ids
            gpu_ids = current_ids
        else:
            births = np.arange(self.next_uid, self.next_uid + birth_count, dtype=np.int64)
            ids = np.concatenate((self.ids if all_survive else self.ids[rows], births))
            gpu_ids = self._upload(ids)
        output = self.slots[1-self.slot]
        n = len(ids)
        gpu_raw = {}
        for name, bank in self.raw_attribute_bank.items():
            value = gpu_capture.get(name)
            if value is None:
                prior = (self._portable_raw[name] if all_survive else
                         self._portable_raw[name].index_select(0, decoded.survivor_rows))
                if torch.equal(raw[name][:len(rows)].view(torch.int32), prior.view(torch.int32)):
                    value = (gpu_reference[name] if all_survive else
                             gpu_reference[name].index_select(0, decoded.survivor_rows.to(self.device)))
                    if birth_count:
                        value = torch.cat((value, raw[name][len(rows):].to(self.device)))
                    elif all_survive:
                        # Neither the values nor their stable IDs changed.
                        gpu_raw[name] = value
                        continue
                else:
                    value = raw[name].to(self.device)
            elif birth_count and len(value) == len(rows):
                value = torch.cat((value, raw[name][len(rows):].to(self.device)))
            if len(value) != n:
                raise ValueError(f'GPU {name} count differs from decoded frame')
            if dense_ids:
                bank[:n].copy_(value)
            else:
                bank.index_copy_(0, gpu_ids, value)
            gpu_raw[name] = value
        activated = dict(rotation=F.normalize(gpu_raw['rotation'], dim=-1),
                         scaling=gpu_raw['logscale'].exp(), opacity=gpu_raw['opacity'].sigmoid(),
                         features=torch.cat((gpu_raw['sh_dc'], gpu_raw['sh_rest']), dim=1))
        for name, value in activated.items():
            if dense_ids:
                self.attribute_bank[name][:n].copy_(value)
            else:
                self.attribute_bank[name].index_copy_(0, gpu_ids, value)
        output['ids'][:n].copy_(gpu_ids)
        gpu_xyz = gpu_capture.get('xyz')
        output['xyz'][:n].copy_(gpu_xyz if gpu_xyz is not None else raw['xyz'])
        if dense_ids:
            self.xyz_bank[:n].copy_(output['xyz'][:n])
        else:
            self.xyz_bank.index_copy_(0, gpu_ids, output['xyz'][:n])
        output['owner'][:n].copy_(decoded.owner)
        output['paths'][:n].copy_(decoded.spatial_paths)
        self.ids, self.next_uid = ids, self.next_uid + birth_count
        self._dense_ids = dense_ids
        self.slot, self.frame = 1-self.slot, frame
        self.expected_digest = decoded_digest
        self.last_open = decoded.spatial_open.clone()
        self.last_sources = decoded.birth_sources.numpy().copy()
        self.last_survivors = rows.copy()
        self._portable_state, self._portable_raw = decoded, raw
        self.view = self._view()
        return self.view


    def _view(self):
        slot=self.slots[self.slot];n=len(self.ids);ids=slot['ids'][:n]
        values=dict(xyz=slot['xyz'][:n])
        for name,bank in self.attribute_bank.items():
            if self._dense_ids:
                slot[name][:n].copy_(bank[:n])
                values[name] = slot[name][:n]
            else:
                values[name]=torch.index_select(bank,0,ids,out=slot[name][:n])
        return StreamView(values,self.sh_degree)

    @torch.no_grad()
    def capture_raw(self):
        slot=self.slots[self.slot]
        actual_ids=slot['ids'][:len(self.ids)].cpu().numpy()
        if not np.array_equal(actual_ids,self.ids):
            raise ValueError('GPU survivor IDs differ from independent host mapping')
        ids = slot['ids'][:len(self.ids)]
        raw = {name: bank.index_select(0, ids).cpu().clone()
               for name, bank in self.raw_attribute_bank.items()}
        raw['xyz']=slot['xyz'][:len(self.ids)].cpu().clone()
        return {name:raw[name] for name in RAW_FIELDS}

    @torch.no_grad()
    def capture_state(self, raw=None):
        raw=self.capture_raw() if raw is None else raw
        slot=self.slots[self.slot];n=len(self.ids)
        return DecodedState(raw['xyz'],raw['rotation'],slot['owner'][:n].cpu().clone(),
            self.reference.parents.clone(),self.frame,torch.from_numpy(self.last_sources.copy()),
            slot['paths'][:n].cpu().clone(),self.last_open.clone(),self.reference.spatial_centers,
            self.reference.spatial_support_ball,self.reference.spatial_leaf_centers,
            self.reference.spatial_leaf_nodes,torch.from_numpy(self.last_survivors.copy()))

    @torch.no_grad()
    def verify_render_inputs(self, raw, *, atol=0.):
        expected=dict(xyz=raw['xyz'].to(self.device),
            rotation=F.normalize(raw['rotation'].to(self.device),dim=-1),
            scaling=raw['logscale'].to(self.device).exp(),
            opacity=raw['opacity'].to(self.device).sigmoid(),
            features=torch.cat((raw['sh_dc'].to(self.device),raw['sh_rest'].to(self.device)),1))
        errors = {}
        for name,value in expected.items():
            actual=self.view.values[name]
            error=float((actual-value).abs().max()) if value.numel() else 0.
            errors[name] = error
            if (not bool(torch.isfinite(actual).all()) or
                    (not torch.equal(actual.view(torch.int32),value.view(torch.int32)) if atol == 0 else error > atol)):
                raise ValueError(f'GPU activated {name} differs from the canonical renderer input: {error}')
        return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stream_dir', type=Path)
    parser.add_argument('--reference', type=Path,
                        help='Reference NPZ; defaults to the first reference in stream_dir')
    parser.add_argument('--through', type=int,
                        help='Last frame to decode; defaults to the last motion file')
    parser.add_argument('--save-npz', type=Path,
                        help='Optionally write decoded raw GS attributes')
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda',
                        help='Use GPU playback by default; cpu selects the reference decoder')
    parser.add_argument('--verify-gpu', action='store_true',
                        help='Compare each GPU state to the CPU reference (copies data back)')
    args = parser.parse_args()
    directory = args.stream_dir
    references = sorted(directory.glob('reference_frame*.npz'))
    reference = args.reference or (references[0] if references else None)
    if reference is None:
        parser.error('No decoder reference found')
    motions = sorted(directory.glob('motion_*.bin'))
    if args.through is None and not motions:
        parser.error('No motion frames found; pass --through for reference only')
    through = args.through if args.through is not None else int(motions[-1].stem[-3:])
    if args.device == 'cpu':
        if args.verify_gpu:
            parser.error('--verify-gpu requires --device cuda')
        state, raw = replay(directory, reference, through)
    else:
        with np.load(reference, allow_pickle=False) as archive:
            raw = {name: torch.from_numpy(archive[name].copy()) for name in RAW_FIELDS}
            state = DecodedState.from_reference(archive)
        if through < state.frame_index:
            parser.error('Requested frame precedes the decoder reference')
        capacity = len(state.xyz)
        for frame in range(state.frame_index + 1, through + 1):
            manifest = json.loads((directory / f'frame_{frame:03d}.json').read_text())
            capacity += int(manifest['birth_count'])
        player = GPUStream(state, raw, capacity=capacity)
        for frame in range(state.frame_index + 1, through + 1):
            player.advance_frame(directory, frame)
            if args.verify_gpu:
                actual = player.capture_raw()
                for name in RAW_FIELDS:
                    error = float((actual[name] - player._portable_raw[name]).abs().max()) if actual[name].numel() else 0.
                    if not bool(torch.isfinite(actual[name]).all()) or error > 1e-5:
                        raise ValueError(f'GPU {name} differs from CPU reference at frame {frame}: {error}')
        raw = player.capture_raw()
        state = player.capture_state(raw)
    if args.save_npz:
        args.save_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.save_npz,
            **{name: value.cpu().numpy() for name, value in raw.items()})
    if args.device == 'cuda':
        print(f'frame={state.frame_index} gs={len(state.xyz)} gpu_state_sha256={state.digest().hex()} '
              f'cpu_reference_sha256={player._portable_state.digest().hex()}')
    else:
        print(f'frame={state.frame_index} gs={len(state.xyz)} decoder_sha256={state.digest().hex()}')


if __name__ == '__main__':
    main()
