"""One trainable newborn pool with stable UID leaves and exact Adam ages.

Correspondence/support data fixes position, quaternion, scale and SH1. Only
DC and opacity learn. Each UID has two small leaves in ONE ordinary Adam, so
new UIDs start at step zero without resetting an older UID's moments/age.
Physical six-field output never depends on an entropy predictor's grid.
"""
import copy

import torch
from torch import nn
from torch.nn import functional as F

from .gaussian import BirthGaussianModel
from .birth_quantization import RAW_FIELDS


C0=.28209479177387814
C1=.4886025119029199
MATURITY_UPDATES=50


def _clone(value,device=None):
    if torch.is_tensor(value):return value.detach().to(device=device or value.device).clone()
    if isinstance(value,dict):return {k:_clone(v,device) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return type(value)(_clone(v,device) for v in value)
    return copy.deepcopy(value)


class _Physical(torch.autograd.Function):
    @staticmethod
    def forward(ctx,parameter,value):return value.detach().clone()
    @staticmethod
    def backward(ctx,gradient):return gradient,None


class PersistentNewbornModel(BirthGaussianModel):
    def __init__(self,sh_degree):
        if sh_degree not in (0,1):
            raise NotImplementedError('Persistent newborns support active SH0/SH1 only')
        super().__init__(sh_degree)
        self.uids=()
        self._records={}
        self._dc={};self._alpha={}
        self.old_rows=torch.empty(0,dtype=torch.long)
        self.frame_index=0
        self._training_args=None
        self._pending_optimizer={};self._pending_groups={}
        self.source_rows=None
        self.active_sh_degree=sh_degree

    @property
    def _features_dc(self):
        return torch.stack([self._dc[u] for u in self.uids]) if getattr(self,'uids',()) else self._dc_seed
    @_features_dc.setter
    def _features_dc(self,value):self._dc_seed=value

    @property
    def _opacity(self):
        return torch.stack([self._alpha[u] for u in self.uids]) if getattr(self,'uids',()) else self._alpha_seed
    @_opacity.setter
    def _opacity(self,value):self._alpha_seed=value

    @property
    def fresh_mask(self):return self.old_rows<0

    @property
    def born_frames(self):
        return torch.tensor([self._records[u]['born_frame'] for u in self.uids],device=self._xyz.device,dtype=torch.long)

    @property
    def age_frames(self):return self.frame_index-self.born_frames+1

    @staticmethod
    def _payload(state):
        return state.get('persistent_newborn',state) if state is not None else None

    def _check_raw(self,raw,device,dtype):
        shapes=dict(xyz=(3,),rotation=(4,),logscale=(3,),opacity=(1,),sh_dc=(1,3),
                    sh_rest=(3 if self.max_sh_degree else 0,3))
        result={k:torch.as_tensor(raw[k],device=device,dtype=dtype).detach().clone() for k in RAW_FIELDS}
        if any(result[k].shape!=shapes[k] or not bool(torch.isfinite(result[k]).all()) for k in RAW_FIELDS):
            raise ValueError('Newborn specifications require finite unbatched six raw fields')
        if not bool(result['rotation'].square().sum()>0):raise ValueError('A fixed quaternion cannot be zero')
        return result

    @staticmethod
    def _bounds(dc,rest):
        a=.5+C0*dc.double()[0]
        norm=C1*rest.double().square().sum(0).sqrt()
        lo=torch.minimum(a-norm,torch.zeros_like(a));hi=torch.maximum(a+norm,torch.ones_like(a))
        lower=((lo+norm-.5)/C0)[None].to(dc)
        upper=((hi-norm-.5)/C0)[None].to(dc)
        return torch.minimum(lower,dc),torch.maximum(upper,dc)

    def install(self,specs,uids=None,previous_state=None,old_rows=None,frame_index=None,device=None,dtype=None):
        """Install/reorder one pool; retained UID leaves stay the same objects.

        Specs rebind fixed geometry for existing UIDs, not their material or
        prior. Use rebind() to synchronize actual committed DC/alpha explicitly.
        previous_state is bank_state(), with optimizer entries keyed by UID.
        """
        specs=list(specs);previous=self._payload(previous_state)
        byuid={s['uid']:s for s in specs}
        if len(byuid)!=len(specs):raise ValueError('Newborn UIDs must be unique')
        if uids is None:
            uids=list(byuid) if specs else list(previous['uids'] if previous is not None else self.uids)
        uids=tuple(uids)
        if len(set(uids))!=len(uids) or any(not isinstance(u,str) or not u for u in uids):
            raise ValueError('Newborn UIDs must be unique nonempty strings')
        saved={} if previous is None else previous['records']
        template=(next(iter(self._dc.values())) if self._dc else
            torch.as_tensor(specs[0]['raw']['xyz']) if specs else
            next(iter(saved.values()))['raw']['xyz'] if saved else torch.empty(0,3))
        device=torch.device(device or template.device)
        dtype=dtype or (template.dtype if template.is_floating_point() else torch.float32)
        if frame_index is not None:self.frame_index=int(frame_index)
        elif previous is not None:self.frame_index=int(previous.get('frame_index',self.frame_index))
        rows=(None if old_rows is None or isinstance(old_rows,dict) else
              torch.as_tensor(old_rows,dtype=torch.long).flatten().tolist())
        if rows is not None and len(rows)!=len(uids):raise ValueError('Old-row map must align with UIDs')
        records={};dcs={};alphas={}
        for i,uid in enumerate(uids):
            spec=byuid.get(uid);prior=saved.get(uid,self._records.get(uid))
            if prior is None and spec is None:raise ValueError(f'No geometry/material record for UID {uid}')
            if prior is None:
                if spec.get('is_original',False):raise ValueError('Original shared Gaussians cannot enter the newborn pool')
                raw=self._check_raw(spec['raw'],device,dtype)
                born=int(spec.get('born_frame',max(1,self.frame_index)))
                if born<1:raise ValueError('Only dynamic-frame newborns can enter this pool')
                low,high=self._bounds(raw['sh_dc'],raw['sh_rest'])
                record=dict(raw=raw,prior_dc=raw['sh_dc'].clone(),lower=low,upper=high,
                    birth_opacity=raw['opacity'].clone(),born_frame=born,old_row=int(spec.get('old_row',-1)),
                    metadata=_clone({k:v for k,v in spec.items() if k not in ('uid','raw','born_frame','old_row')}))
                dcvalue,alphavalue=raw['sh_dc'],raw['opacity']
            else:
                record=_clone(prior,device)
                record['raw']=self._check_raw(record['raw'],device,dtype)
                for key in ('prior_dc','lower','upper','birth_opacity'):
                    record[key]=record[key].to(device=device,dtype=dtype)
                dcvalue=torch.as_tensor(record.get('dc_parameter',record['raw']['sh_dc']),device=device,dtype=dtype)
                alphavalue=torch.as_tensor(record.get('opacity_parameter',record['raw']['opacity']),device=device,dtype=dtype)
                if uid in self._dc and previous is None:
                    dcvalue=self._dc[uid].detach();alphavalue=self._alpha[uid].detach()
                if spec is not None:
                    current=self._check_raw(spec['raw'],device,dtype)
                    for key in ('xyz','rotation','logscale'):record['raw'][key]=current[key]
                    record['metadata'].update(_clone({k:v for k,v in spec.items() if k not in ('uid','raw','born_frame','old_row')}))
            if old_rows is not None:record['old_row']=int(old_rows[uid]) if isinstance(old_rows,dict) else int(rows[i])
            elif spec is not None and 'old_row' in spec:record['old_row']=int(spec['old_row'])
            if uid in self._dc:
                if self._dc[uid].device!=device or self._dc[uid].dtype!=dtype:
                    raise ValueError('Restore onto a new model to change a live pool device/dtype')
                dcs[uid],alphas[uid]=self._dc[uid],self._alpha[uid]
                if previous is not None:
                    with torch.no_grad():dcs[uid].copy_(dcvalue);alphas[uid].copy_(alphavalue)
            else:
                dcs[uid]=nn.Parameter(dcvalue.detach().clone());alphas[uid]=nn.Parameter(alphavalue.detach().clone())
            records[uid]=record
        self.uids,self._records,self._dc,self._alpha=uids,records,dcs,alphas
        if previous is not None:
            self._pending_optimizer={u:_clone(previous.get('optimizer',{}).get(u,{})) for u in uids}
            self._pending_groups=_clone(previous.get('optimizer_groups',{}))
        else:self._pending_optimizer={u:v for u,v in self._pending_optimizer.items() if u in records}
        self._sync_fixed(device,dtype)
        self._refresh_optimizer()
        if self.optimizer is not None:self._load_pending_optimizer()
        self.project_parameters()
        return self

    def _sync_fixed(self,device=None,dtype=None):
        device=device or self._xyz.device;dtype=dtype or self._xyz.dtype
        shapes=dict(xyz=(3,),rotation=(4,),logscale=(3,),sh_rest=(3 if self.max_sh_degree else 0,3))
        attrs=dict(xyz='_xyz',rotation='_rotation',logscale='_scaling',sh_rest='_features_rest')
        for key,shape in shapes.items():
            value=(torch.stack([self._records[u]['raw'][key] for u in self.uids]) if self.uids else
                   torch.empty((0,*shape),device=device,dtype=dtype))
            setattr(self,attrs[key],value.detach().clone())
        self._dc_seed=torch.empty(0,1,3,device=device,dtype=dtype)
        self._alpha_seed=torch.empty(0,1,device=device,dtype=dtype)
        self.old_rows=torch.tensor([self._records[u]['old_row'] for u in self.uids],device=device,dtype=torch.long)
        positive=self.old_rows[self.old_rows>=0]
        if len(positive.unique())!=len(positive):raise ValueError('Two newborn UIDs cannot own the same old row')
        self.max_radii2D=torch.zeros(len(self.uids),device=device,dtype=dtype)
        self.xyz_gradient_accum=torch.zeros(len(self.uids),1,device=device,dtype=dtype)
        self.denom=torch.zeros_like(self.xyz_gradient_accum)

    def _step(self,uid,name):
        p=self._dc[uid] if name=='f_dc' else self._alpha[uid]
        state=(self.optimizer.state.get(p,{}) if self.optimizer is not None else
               self._pending_optimizer.get(uid,{}).get(name,{}))
        value=state.get('step',0)
        return int(value.item() if torch.is_tensor(value) else value)

    def uid_parameters(self,uid):return dict(f_dc=self._dc[uid],opacity=self._alpha[uid])

    def intrinsic_dc_bounds(self):
        if not self.uids:return self._dc_seed.clone(),self._dc_seed.clone()
        return (torch.stack([self._records[u]['lower'] for u in self.uids]),
                torch.stack([self._records[u]['upper'] for u in self.uids]))

    def _physical_dc(self):
        value=self._features_dc
        if not self.uids:return value
        low,high=self.intrinsic_dc_bounds()
        prior=torch.stack([self._records[u]['prior_dc'] for u in self.uids])
        target=torch.maximum(low,torch.minimum(value,high))
        target=torch.where(value.detach()==prior,prior,target)
        return _Physical.apply(value,target)

    def _physical_opacity(self):
        value=self._opacity
        if not self.uids:return value
        protected=torch.tensor([self._step(u,'opacity')<MATURITY_UPDATES for u in self.uids],device=value.device)
        floor=torch.stack([self._records[u]['birth_opacity'] for u in self.uids])
        target=torch.where(protected[:,None],torch.maximum(value,floor),value)
        return _Physical.apply(value,target)

    def raw(self):
        return dict(xyz=self._xyz,rotation=self._rotation,logscale=self._scaling,
                    sh_dc=self._physical_dc(),sh_rest=self._features_rest,opacity=self._physical_opacity())

    def birth_parameter_value(self,name):return self.raw()[name]
    def persistent_quantized_raw(self,config,extent,reference,source_rows):return self.raw()
    @property
    def get_xyz(self):return self._xyz
    @property
    def get_rotation(self):return F.normalize(self._rotation,dim=-1)
    @property
    def get_scaling(self):return self._scaling.exp()
    @property
    def get_opacity(self):return self._physical_opacity().sigmoid()
    @property
    def get_features(self):return torch.cat((self._physical_dc(),self._features_rest),1)

    def training_setup(self,args):
        self._training_args=args;self.percent_dense=args.percent_dense
        if self.optimizer is None:
            self.optimizer=torch.optim.Adam([
                dict(params=[self._dc[u] for u in self.uids],lr=args.feature_lr,name='f_dc'),
                dict(params=[self._alpha[u] for u in self.uids],lr=args.opacity_lr,name='opacity')],
                lr=0.,eps=1.e-15,foreach=True)
        else:self._refresh_optimizer()
        self._load_pending_optimizer()
        return self

    initialize_incremental_optimizer=training_setup

    def _refresh_optimizer(self):
        if self.optimizer is None:return
        byname=dict(f_dc=[self._dc[u] for u in self.uids],opacity=[self._alpha[u] for u in self.uids])
        keep=set(byname['f_dc']+byname['opacity'])
        for group in self.optimizer.param_groups:group['params']=byname[group['name']]
        for parameter in list(self.optimizer.state):
            if parameter not in keep:del self.optimizer.state[parameter]

    def _load_pending_optimizer(self):
        if self.optimizer is None:return
        for group in self.optimizer.param_groups:
            saved=self._pending_groups.get(group['name'],{})
            group.update({k:_clone(v) for k,v in saved.items() if k!='params'})
        for uid,snapshot in self._pending_optimizer.items():
            if uid not in self._dc:continue
            for name,p in self.uid_parameters(uid).items():
                state=snapshot.get(name,{})
                self.optimizer.state.pop(p,None)
                if state:
                    self.optimizer.state[p]={k:(v.detach().cpu().clone() if k=='step' and torch.is_tensor(v) else
                        v.detach().to(device=p.device,dtype=p.dtype).clone() if torch.is_tensor(v) else _clone(v))
                        for k,v in state.items()}
        self._pending_optimizer={};self._pending_groups={}

    @torch.no_grad()
    def project_parameters(self):
        if self.uids:
            dc=self._physical_dc();opacity=self._physical_opacity()
            torch._foreach_copy_([self._dc[u] for u in self.uids],list(dc.unbind(0)))
            torch._foreach_copy_([self._alpha[u] for u in self.uids],list(opacity.unbind(0)))
        return self

    project_=project_parameters

    @torch.no_grad()
    def rebind(self,uids,raw,old_rows=None,frame_index=None):
        """Bind current fixed geometry and actually committed material, keep Adam.

        Raw quaternions and fixed SH1 are never renormalized/relearned. A changed
        SH1 is an error rather than an implicit material-prior replacement.
        """
        uids=tuple(uids)
        batched='xyz' in raw
        if batched and len(uids)==1 and torch.as_tensor(raw['xyz']).ndim==1:
            raw={k:torch.as_tensor(v)[None] for k,v in raw.items()}
        for i,uid in enumerate(uids):
            if uid not in self._records:raise KeyError(uid)
            item={k:raw[k][i] for k in RAW_FIELDS} if batched else raw[uid]
            value=self._check_raw(item,self._xyz.device,self._xyz.dtype)
            if not torch.equal(value['sh_rest'],self._records[uid]['raw']['sh_rest']):
                raise ValueError('An established UID cannot change its fixed physical SH1')
            for key in ('xyz','rotation','logscale'):self._records[uid]['raw'][key]=value[key]
            self._dc[uid].copy_(value['sh_dc']);self._alpha[uid].copy_(value['opacity'])
            if old_rows is not None:self._records[uid]['old_row']=int(old_rows[uid] if isinstance(old_rows,dict) else old_rows[i])
        if frame_index is not None:self.frame_index=int(frame_index)
        self._sync_fixed();return self

    rebind_geometry=rebind

    def bank_state(self,device='cpu'):
        raw=self.raw();records={};optimizer={}
        for i,uid in enumerate(self.uids):
            record=_clone(self._records[uid],device)
            record['raw']={k:_clone(v[i],device) for k,v in raw.items()}
            record['dc_parameter']=_clone(self._dc[uid],device)
            record['opacity_parameter']=_clone(self._alpha[uid],device)
            records[uid]=record
            optimizer[uid]={name:_clone(self.optimizer.state.get(p,{}),'cpu') for name,p in self.uid_parameters(uid).items()} if self.optimizer is not None else _clone(self._pending_optimizer.get(uid,{}),'cpu')
        groups=({g['name']:{k:_clone(v,'cpu') for k,v in g.items() if k!='params'} for g in self.optimizer.param_groups}
                if self.optimizer is not None else _clone(self._pending_groups,'cpu'))
        return dict(version=1,sh_degree=self.max_sh_degree,uids=list(self.uids),frame_index=self.frame_index,
                    records=records,optimizer=optimizer,optimizer_groups=groups)

    def capture(self):return dict(persistent_newborn=self.bank_state(),source_rows=_clone(self.source_rows,'cpu'))

    def restore(self,packet,training_args=None,device=None):
        state=self._payload(packet)
        if state['sh_degree']!=self.max_sh_degree:raise ValueError('Persistent SH degree changed')
        self.install([],previous_state=state,device=device)
        if training_args is not None:self.training_setup(training_args)
        self.source_rows=_clone(packet.get('source_rows'),self._xyz.device)
        return self

    def _selection_uids(self,selection):
        if isinstance(selection,str):return (selection,)
        if isinstance(selection,(list,tuple)) and (not selection or isinstance(selection[0],str)):
            ids=tuple(selection)
        else:
            value=torch.as_tensor(selection)
            indexes=value.nonzero().flatten().tolist() if value.dtype==torch.bool else value.long().flatten().tolist()
            if value.dtype==torch.bool and len(value)!=len(self.uids):raise ValueError('Subset mask length changed')
            ids=tuple(self.uids[i] for i in indexes)
        if len(set(ids))!=len(ids) or any(u not in self._records for u in ids):raise ValueError('Invalid UID selection')
        return ids

    def subset(self,selection):
        uids=self._selection_uids(selection)
        result=type(self)(self.max_sh_degree).install([],uids=uids,previous_state=self.bank_state(),
            device=self._xyz.device,dtype=self._xyz.dtype)
        result.spatial_lr_scale=self.spatial_lr_scale
        if self._training_args is not None:result.training_setup(self._training_args)
        if self.source_rows is not None:
            ids=torch.tensor([self.uids.index(u) for u in uids],device=self.source_rows.device,dtype=torch.long)
            result.source_rows=self.source_rows[ids].clone()
        return result

    persistent_subset=subset
    subset_by_uid=subset

    def commit(self,uids,old_rows=None):
        """Retain chosen UID leaves in place; the controller supplies layout rows."""
        uids=self._selection_uids(uids)
        self.uids=uids;self._records={u:self._records[u] for u in uids}
        self._dc={u:self._dc[u] for u in uids};self._alpha={u:self._alpha[u] for u in uids}
        if old_rows is not None:
            for i,u in enumerate(uids):self._records[u]['old_row']=int(old_rows[u] if isinstance(old_rows,dict) else old_rows[i])
        self._pending_optimizer={u:v for u,v in self._pending_optimizer.items() if u in self._records}
        self._sync_fixed();self._refresh_optimizer();self.source_rows=None
        return self

    def rollback_uids(self,uids,previous_state):
        """Restore rejected active UID values AND moments; discard rejected fresh UIDs."""
        previous=self._payload(previous_state);remove=[]
        for uid in uids:
            if uid not in previous['records']:remove.append(uid);continue
            if uid not in self._records:raise KeyError(uid)
            record=_clone(previous['records'][uid],self._xyz.device)
            self._records[uid]=record
            with torch.no_grad():
                self._dc[uid].copy_(record.get('dc_parameter',record['raw']['sh_dc']))
                self._alpha[uid].copy_(record.get('opacity_parameter',record['raw']['opacity']))
            self._pending_optimizer[uid]=_clone(previous.get('optimizer',{}).get(uid,{}),'cpu')
        if remove:self.commit([u for u in self.uids if u not in remove])
        self._sync_fixed()
        if self.optimizer is not None:self._load_pending_optimizer()
        return self

    def prune_points(self,remove):
        value=torch.as_tensor(remove)
        mask=value if value.dtype==torch.bool else torch.zeros(len(self.uids),dtype=torch.bool)
        if value.dtype!=torch.bool:mask[value.long()]=True
        return self.commit([u for i,u in enumerate(self.uids) if not bool(mask[i])])

    @staticmethod
    def merge_bank_states(previous_state,updated_state,uids,raw,old_rows,frame_index=None):
        """CPU merge of active updates with an arbitrarily larger inactive bank.

        uids/raw/old_rows describe ALL surviving newborns in the actual committed
        layout, excluding the protected original population. No GPU Parameters
        are built for inactive records. Rejected active updates should already
        have been rolled back in updated_state before this merge.
        """
        previous=PersistentNewbornModel._payload(previous_state)
        updated=PersistentNewbornModel._payload(updated_state)
        base=updated if updated is not None else previous
        if base is None:raise ValueError('A bank merge requires a previous or updated bank')
        uids=tuple(uids)
        if len(set(uids))!=len(uids):raise ValueError('Live bank UIDs must be unique')
        batched='xyz' in raw
        values=_clone(raw,'cpu')
        rows=([int(old_rows[u]) for u in uids] if isinstance(old_rows,dict) else
              torch.as_tensor(old_rows,dtype=torch.long).flatten().tolist())
        if len(rows)!=len(uids) or any(r<0 for r in rows) or len(set(rows))!=len(rows):
            raise ValueError('A committed bank requires one distinct actual layout row per UID')
        helper=PersistentNewbornModel(base['sh_degree'])
        records={};optimizer={}
        for i,uid in enumerate(uids):
            source=(updated if updated is not None and uid in updated['records'] else previous)
            if source is None or uid not in source['records']:raise KeyError(uid)
            record=_clone(source['records'][uid],'cpu')
            item={k:values[k][i] for k in RAW_FIELDS} if batched else values[uid]
            dtype=torch.as_tensor(item['xyz']).dtype
            actual=helper._check_raw(item,torch.device('cpu'),dtype)
            if not torch.equal(actual['sh_rest'],record['raw']['sh_rest'].to(actual['sh_rest'])):
                raise ValueError('Committed layout changed an established UID physical SH1')
            record['raw']=actual;record['old_row']=rows[i]
            record['dc_parameter']=actual['sh_dc'].clone()
            record['opacity_parameter']=actual['opacity'].clone()
            records[uid]=record
            optimizer[uid]=_clone(source.get('optimizer',{}).get(uid,{}),'cpu')
        return dict(version=1,sh_degree=base['sh_degree'],uids=list(uids),
            frame_index=int(base.get('frame_index',0) if frame_index is None else frame_index),
            records=records,optimizer=optimizer,optimizer_groups=_clone(base.get('optimizer_groups',{}),'cpu'))
