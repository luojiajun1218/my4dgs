"""Incremental GPU materialization of the active translation/offset stream.

Packets retain their original bytes and checksums. Playback checks declared
geometry digest links and exact CPU opacity predictor hashes. Full GPU geometry
and raw-state hashes are checked by the separate independent verification pass.
No decoded future frames are cached.
"""
import hashlib
import struct

import numpy as np
import torch
from torch.nn import functional as F

from .birth_quantization import RAW_FIELDS, physical_covariance
from .legacy_opacity_codec import (canonical_alpha_logits, _canonical_alpha_logits,
                                   _canonical_saturated_alpha_logits)
from .rigid_residual_tree import quaternion_multiply
from .rigid_tree_codec import DecodedState, _validate_state
from .stream_cuda import backend


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
        self.parents=reference_state.parents.numpy().copy()
        self.depth=reference_state.spatial_paths.shape[1]
        self.regions=len(self.parents)
        self.rest=raw['sh_rest'].shape[1]
        if self.rest not in (0,3):
            raise ValueError('GPU stream currently supports SH0/SH1')
        count=len(reference_state.xyz)
        shapes=dict(xyz=(count,3),rotation=(count,4),logscale=(count,3),opacity=(count,1),
                    sh_dc=(count,1,3),sh_rest=(count,self.rest,3))
        if any(raw[name].dtype != torch.float32 or tuple(raw[name].shape) != shape or
               not bool(torch.isfinite(raw[name]).all()) for name,shape in shapes.items()):
            raise ValueError('Reference raw attributes need aligned finite float32 fields')
        self.device=torch.device(device)
        self.capacity=int(capacity)
        self.ids=np.arange(len(reference_state.xyz),dtype=np.int64)
        self.next_uid=len(self.ids)
        self.frame=0
        self.expected_digest=reference_state.digest()
        self.host={name:np.empty((capacity,*raw[name].shape[1:]),dtype=np.float32)
                   for name in RAW_FIELDS if name != 'xyz'}
        for name,array in self.host.items():
            array[:len(self.ids)]=raw[name].numpy()
        self.pending_q=self.ids.copy()
        self.xyz_bank=torch.empty((capacity,3),device=self.device)
        self.far_rows=torch.empty(capacity,device=self.device,dtype=torch.long)
        self.far_count=torch.empty(1,device=self.device,dtype=torch.int32)
        self.xyz_bank[:len(self.ids)].copy_(raw['xyz'])
        self.attribute_bank={name:torch.empty(shape,device=self.device) for name,shape in
            dict(rotation=(capacity,4),scaling=(capacity,3),opacity=(capacity,1),
                 features=(capacity,1+self.rest,3)).items()}
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
        self.planes=self._upload(reference_state.spatial_centers.numpy())
        ball=reference_state.spatial_support_ball
        leaves=reference_state.spatial_leaf_centers
        nodes=reference_state.spatial_leaf_nodes
        self.ball=self._upload(np.empty(0,np.float64) if ball is None else ball.numpy())
        self.leaves=self._upload(np.empty((0,3),np.float64) if leaves is None else leaves.double().numpy())
        self.nodes=self._upload(np.empty(0,np.int64) if nodes is None else nodes.numpy())
        self.last_open=reference_state.spatial_open.clone()
        self.last_sources=np.empty(0,np.int64)
        self.last_survivors=np.arange(len(self.ids),dtype=np.int64)
        self._activate(self.ids,all_fields=True)
        self.view=self._view()
        backend()  # Build before any measured playback pass.

    def _upload(self, array):
        tensor=torch.from_numpy(np.ascontiguousarray(array))
        # PyTorch's pinned allocator records asynchronous copy lifetimes.
        return tensor.pin_memory().to(self.device,non_blocking=True)

    def _activate(self, ids, *, all_fields=False):
        if not len(ids): return
        gpu_ids=self._upload(ids)
        rotation=F.normalize(self._upload(self.host['rotation'][ids]),dim=-1)
        self.attribute_bank['rotation'].index_copy_(0,gpu_ids,rotation)
        if all_fields:
            self.attribute_bank['scaling'].index_copy_(0,gpu_ids,self._upload(self.host['logscale'][ids]).exp())
            self.attribute_bank['opacity'].index_copy_(0,gpu_ids,self._upload(self.host['opacity'][ids]).sigmoid())
            features=np.concatenate((self.host['sh_dc'][ids],self.host['sh_rest'][ids]),1)
            self.attribute_bank['features'].index_copy_(0,gpu_ids,self._upload(features))

    def _canonicalize_old_q(self):
        if not len(self.pending_q): return
        ids=self.pending_q
        previous=self.host['rotation'][ids]
        value=torch.from_numpy(previous.copy())
        identity=torch.zeros_like(value);identity[:,0]=1.
        current=quaternion_multiply(identity,value).numpy()
        following=quaternion_multiply(identity,torch.from_numpy(current)).numpy()
        changed=np.any(current.view(np.uint32)!=previous.view(np.uint32),axis=1)
        unstable=np.any(following.view(np.uint32)!=current.view(np.uint32),axis=1)
        self.host['rotation'][ids]=current
        self.pending_q=ids[unstable]
        self._activate(ids[changed])

    def _opacity(self, update, old_ids):
        if update is None: return
        reference=self.host['opacity'][old_ids]
        digest=hashlib.sha256(struct.pack('<I',len(reference))+reference.astype('<f4',copy=False).tobytes()).digest()
        if update.count != len(old_ids) or digest != update.prior:
            raise ValueError('Exact opacity predictor hash mismatch')
        ids=old_ids[update.rows]
        selected=torch.from_numpy(self.host['opacity'][ids].copy())
        symbols=torch.from_numpy(update.values.copy())
        if update.mode=='alpha_canonical':
            logits=canonical_alpha_logits(selected,symbols,update.step)
        elif update.mode=='alpha':
            logits=_canonical_alpha_logits(selected,symbols,update.step)
        elif update.mode=='saturated_alpha':
            logits=_canonical_saturated_alpha_logits(selected,symbols,update.step)
        else:
            logits=selected+symbols.float()*update.step
        if not bool(torch.isfinite(logits).all()):
            raise ValueError('Nonfinite decoded opacity')
        self.host['opacity'][ids]=logits.numpy()
        self.attribute_bank['opacity'].index_copy_(0,self._upload(ids),self._upload(logits.numpy()).sigmoid())

    @torch.no_grad()
    def advance(self, step):
        if (step.frame != self.frame+1 or step.prior_digest != self.expected_digest or
                step.previous_count != len(self.ids)):
            raise ValueError('Motion frame or declared geometry-state link mismatch')
        if self.next_uid+step.birth_count > self.capacity:
            raise ValueError('Stable-ID bank capacity exceeded')
        mask=np.ones(len(self.ids),dtype=bool);mask[step.deleted]=False
        survivor_rows=np.flatnonzero(mask)
        old_ids=self.ids[mask]
        if len(old_ids)!=step.old_count:
            raise ValueError('Survivor count mismatch')
        self._canonicalize_old_q()
        if step.opacity_before_births:
            self._opacity(step.opacity,old_ids)
        birth_ids=np.arange(self.next_uid,self.next_uid+step.birth_count,dtype=np.int64)
        if step.birth_count:
            source_ids=old_ids[step.sources]
            source_q=torch.from_numpy(self.host['rotation'][source_ids].copy())
            q=F.normalize(source_q,dim=-1)+torch.from_numpy(step.pose_symbols[:,3:].copy())*step.pose_steps[1]
            self.host['rotation'][birth_ids]=q.numpy()
            for index,name in enumerate(('logscale','opacity','sh_dc','sh_rest')):
                encoded=step.attributes.values[name]
                if step.attributes.steps is None:
                    value=encoded.astype(np.float32,copy=True)
                else:
                    delta=encoded.astype(np.float32)*np.float32(step.attributes.steps[index])
                    value=self.host[name][source_ids]+delta
                if not bool(np.isfinite(value).all()):
                    raise ValueError('Nonfinite newborn appearance')
                self.host[name][birth_ids]=value
            if not bool(torch.isfinite(q).all()):
                raise ValueError('Nonfinite newborn quaternion')
            self._activate(birth_ids,all_fields=True)
        if not step.opacity_before_births:
            self._opacity(step.opacity,old_ids)
        regional=step.regional_symbols.copy()
        if step.relative:
            start,width=1,8
            while start<len(regional):
                end=min(start+width,len(regional))
                regional[start:end]+=regional[self.parents[start:end]]
                start,width=end,width*8
        regional=regional.astype(np.float32)*np.float32(step.region_step)
        output=self.slots[1-self.slot]
        n=step.old_count+step.birth_count
        ids=output['ids'][:n];xyz=output['xyz'][:n]
        owner=output['owner'][:n];paths=output['paths'][:n]
        gates=self._upload(step.opened)
        adjusted=step.deleted-np.arange(len(step.deleted),dtype=np.int64)
        engine=backend()
        engine.advance_old(self.xyz_bank,self.slots[self.slot]['ids'][:len(self.ids)],
            self._upload(adjusted),self._upload(step.gs_rows),self._upload(step.gs_symbols),
            self._upload(regional),step.gs_step,self.planes,gates,self.ball,self.leaves,self.nodes,
            ids,xyz,owner,paths,step.old_count,self.far_rows,self.far_count)
        if step.birth_count:
            engine.append_births(self.xyz_bank,ids,self._upload(step.sources),self._upload(step.pose_symbols),
                step.pose_steps[0],self.next_uid,step.old_count,self.planes,gates,self.ball,self.leaves,
                self.nodes,xyz,owner,paths,self.far_rows,self.far_count)
        self.ids=np.concatenate((old_ids,birth_ids))
        self.next_uid+=step.birth_count
        self.pending_q=np.concatenate((self.pending_q,birth_ids))
        self.slot=1-self.slot
        self.frame=step.frame
        self.expected_digest=step.digest
        self.last_open=torch.from_numpy(step.opened.copy())
        self.last_sources=step.sources.copy()
        self.last_survivors=survivor_rows
        self.view=self._view()
        return self.view

    def _view(self):
        slot=self.slots[self.slot];n=len(self.ids);ids=slot['ids'][:n]
        values=dict(xyz=slot['xyz'][:n])
        for name,bank in self.attribute_bank.items():
            values[name]=torch.index_select(bank,0,ids,out=slot[name][:n])
        return StreamView(values,1 if self.rest else 0)

    @torch.no_grad()
    def capture_raw(self):
        slot=self.slots[self.slot]
        actual_ids=slot['ids'][:len(self.ids)].cpu().numpy()
        if not np.array_equal(actual_ids,self.ids):
            raise ValueError('GPU survivor IDs differ from independent host mapping')
        raw={name:torch.from_numpy(array[self.ids].copy()) for name,array in self.host.items()}
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
    def verify_render_inputs(self, raw):
        expected=dict(xyz=raw['xyz'].to(self.device),
            rotation=F.normalize(raw['rotation'].to(self.device),dim=-1),
            scaling=raw['logscale'].to(self.device).exp(),
            opacity=raw['opacity'].to(self.device).sigmoid(),
            features=torch.cat((raw['sh_dc'].to(self.device),raw['sh_rest'].to(self.device)),1))
        for name,value in expected.items():
            actual=self.view.values[name]
            if not torch.equal(actual.view(torch.int32),value.view(torch.int32)):
                error=float((actual-value).abs().max())
                raise ValueError(f'GPU activated {name} differs from the canonical renderer input: {error}')
