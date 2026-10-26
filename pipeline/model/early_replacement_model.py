"""One-frame early 1:1 renewal with live source pose and a shared Adam.

The ordered renderer replaces original slots, not their coverage. Local pose
residuals keep following source motion until its final pose is detached; that
transition never rebases parameters or resets their optimizer state.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .gaussian import BirthGaussianModel
from .birth_quantization import RAW_FIELDS,physical_covariance
from .conservative_family import _clone_tree
from .reincarnation_model import project_sh1_sphere,sh1_sphere_range


class _Exact(torch.autograd.Function):
    @staticmethod
    def forward(ctx,parameter,target):return target.detach().clone()
    @staticmethod
    def backward(ctx,gradient):return gradient,None


def _multiply(a,b):
    aw,av=a[:,:1],a[:,1:];bw,bv=b[:,:1],b[:,1:]
    return torch.cat((aw*bw-(av*bv).sum(-1,keepdim=True),
        aw*bv+bw*av+torch.cross(av,bv,dim=-1)),-1)


def _rotate(q,value):
    q=F.normalize(q,dim=-1);v=q[:,1:]
    cross=torch.cross(v,value,dim=-1)
    return value+2*(q[:,:1]*cross+torch.cross(v,cross,dim=-1))


def _rotation_vector(value):
    theta=torch.linalg.vector_norm(value,dim=-1,keepdim=True)
    return torch.cat((torch.cos(theta*.5),value*(.5*torch.sinc(theta/(2*math.pi)))),-1)


def _logits(alpha):
    tiny=max(math.exp(-100),torch.finfo(alpha.dtype).tiny)
    bounded=alpha.clamp(tiny,1-torch.finfo(alpha.dtype).eps)
    value=bounded.log()-torch.log1p(-bounded)
    return torch.where(alpha<=0,alpha.new_full((),-100.),
                       torch.where(alpha>=1,alpha.new_full((),100.),value))


class EarlyReplacementModel(BirthGaussianModel):
    GROUPS=(('early_xyz','local_xyz'),('early_rotation','local_rotvec'),
            ('early_scaling','_scaling'),('early_opacity','_opacity'),
            ('early_f_dc','_features_dc'),('early_f_rest','_features_rest'))

    def __init__(self,sh_degree=1):
        if sh_degree not in (0,1):raise NotImplementedError('Early renewal supports active SH0/SH1 only')
        super().__init__(sh_degree)
        self.spatial_lr_scale=1.
        self.parent_rows=torch.empty(0,dtype=torch.long)
        self.local_xyz=nn.Parameter(torch.empty(0,3))
        self.local_rotvec=nn.Parameter(torch.empty(0,3))
        self._seed_raw={};self._seed_alpha=torch.empty(0,1)
        self._source_xyz=torch.empty(0,3);self._source_q=torch.empty(0,4)
        self._parent_max=-1;self.codec_alpha=False;self._saved_groups={}

    def install(self,raw_seed,parent_rows,seed_alpha=None):
        raw={k:torch.as_tensor(raw_seed[k]).detach().clone() for k in RAW_FIELDS}
        xyz=raw['xyz'];n=len(xyz)
        shapes=dict(xyz=(n,3),rotation=(n,4),logscale=(n,3),opacity=(n,1),sh_dc=(n,1,3),
                    sh_rest=(n,3 if self.max_sh_degree else 0,3))
        if any(raw[k].shape!=shapes[k] or not bool(torch.isfinite(raw[k]).all()) for k in RAW_FIELDS):
            raise ValueError('Early renewal needs finite, aligned six raw source fields')
        rows=torch.as_tensor(parent_rows,device=xyz.device,dtype=torch.long).flatten().clone()
        if rows.shape!=(n,) or len(rows.unique())!=n or bool((rows<0).any()):
            raise ValueError('Every early learner must own one unique original source row')
        alpha=(raw['opacity'].sigmoid() if seed_alpha is None else
               torch.as_tensor(seed_alpha,device=xyz.device,dtype=xyz.dtype).detach().clone())
        if alpha.shape!=(n,1) or not bool(torch.isfinite(alpha).all()) or bool(((alpha<0)|(alpha>1)).any()):
            raise ValueError('Source rendered alpha must be finite in [0,1]')
        self._seed_raw=raw;self._seed_alpha=alpha
        self.parent_rows=rows;self.source_rows=rows.clone()
        self._parent_max=int(rows.max()) if n else -1
        self._xyz=raw['xyz'].clone();self._rotation=raw['rotation'].clone()
        self.local_xyz=nn.Parameter(torch.zeros_like(xyz))
        self.local_rotvec=nn.Parameter(torch.zeros_like(xyz))
        self._features_dc=nn.Parameter(raw['sh_dc'].clone())
        self._features_rest=nn.Parameter(raw['sh_rest'].clone())
        self._scaling=nn.Parameter(raw['logscale'].clone())
        # Probability-space alpha keeps even alpha=1 learnable. Its physical
        # raw field is a canonical finite logit, rather than this parameter.
        self._opacity=nn.Parameter(alpha.clone())
        self._source_xyz=xyz.clone();self._source_q=raw['rotation'].clone()
        lo,hi=sh1_sphere_range(raw['sh_dc'],raw['sh_rest'])
        self._low,self._high=lo.clamp_max(0),hi.clamp_min(1)
        self.codec_alpha=False;self.active_sh_degree=self.max_sh_degree
        self.max_radii2D=xyz.new_zeros(n);self.xyz_gradient_accum=xyz.new_zeros(n,1);self.denom=xyz.new_zeros(n,1)
        return self

    def bind_source_pose(self,full_xyz,full_rawq,detach=False):
        if full_xyz.ndim!=2 or full_xyz.shape[1]!=3 or full_rawq.shape!=(len(full_xyz),4) or self._parent_max>=len(full_xyz):
            raise ValueError('Live source pose must address the unchanged original old layout')
        xyz=full_xyz.index_select(0,self.parent_rows)
        q=full_rawq.index_select(0,self.parent_rows)
        self._source_xyz=xyz.detach() if detach else xyz
        self._source_q=q.detach() if detach else q
        return self

    def bind_codec_alpha(self,enabled=True):self.codec_alpha=bool(enabled);return self

    def set_alpha_trainable(self,enabled):
        self._opacity.requires_grad_(bool(enabled))
        if not enabled:self._opacity.grad=None
        return self

    def _world_pose(self):
        xyz=self._source_xyz+_rotate(self._source_q,self.local_xyz)
        zero_xyz=(self.local_xyz.detach()==0).all(-1,keepdim=True)
        xyz=_Exact.apply(xyz,torch.where(zero_xyz,self._source_xyz,xyz))
        q=_multiply(self._source_q,_rotation_vector(self.local_rotvec))
        zero_q=(self.local_rotvec.detach()==0).all(-1,keepdim=True)
        q=_Exact.apply(q,torch.where(zero_q,self._source_q,q))
        return xyz,q

    def _material(self):
        dc,rest,_=project_sh1_sphere(self._features_dc,self._features_rest,self._low,self._high)
        pristine=((self._features_dc.detach()==self._seed_raw['sh_dc']).all((1,2)) &
                  (self._features_rest.detach()==self._seed_raw['sh_rest']).all((1,2)))
        dc=torch.where(pristine[:,None,None],self._seed_raw['sh_dc'],dc)
        rest=torch.where(pristine[:,None,None],self._seed_raw['sh_rest'],rest)
        return _Exact.apply(self._features_dc,dc),_Exact.apply(self._features_rest,rest)

    def _raw_opacity(self):
        alpha=self._opacity.clamp(0,1)
        pristine=self._opacity.detach()==self._seed_alpha
        logits=_logits(alpha)
        return _Exact.apply(logits,torch.where(pristine,self._seed_raw['opacity'],logits))

    def raw(self):
        xyz,q=self._world_pose();dc,rest=self._material()
        return dict(xyz=xyz,rotation=q,logscale=self._scaling,opacity=self._raw_opacity(),sh_dc=dc,sh_rest=rest)

    def birth_parameter_value(self,name):return self.raw()[name]
    def early_quantized_raw(self,config,extent,reference,source_rows):return self.raw()
    @property
    def get_xyz(self):return self._world_pose()[0]
    @property
    def get_rotation(self):return F.normalize(self._world_pose()[1],dim=-1)
    @property
    def get_scaling(self):return self._scaling.exp()
    @property
    def get_features(self):return torch.cat(self._material(),1)
    def get_covariance(self,scaling_modifier=1.):
        return physical_covariance(self.get_scaling,self.get_rotation,scaling_modifier)
    @property
    def get_opacity(self):
        target=self._raw_opacity().sigmoid()
        if not self.codec_alpha:
            target=torch.where(self._opacity.detach()==self._seed_alpha,self._seed_alpha,target)
        return _Exact.apply(self._opacity,target)

    @torch.no_grad()
    def project_parameters(self):
        dc,rest=self._material();self._features_dc.copy_(dc);self._features_rest.copy_(rest)
        self._opacity.clamp_(0,1)
        return self

    project_=project_parameters

    def attach_optimizer(self,sharedAdam,args):
        self.optimizer=sharedAdam;self._training_args=args
        rates=dict(early_xyz=args.position_lr_init*self.spatial_lr_scale,early_rotation=args.rotation_lr,
            early_scaling=args.scaling_lr,early_opacity=args.opacity_lr,
            early_f_dc=args.feature_lr,early_f_rest=args.feature_lr/20)
        groups={g.get('name'):g for g in sharedAdam.param_groups}
        for name,attr in self.GROUPS:
            parameter=getattr(self,attr)
            if name not in groups:
                sharedAdam.add_param_group(dict(params=[parameter],lr=rates[name],name=name,
                                               eps=1e-15,betas=(.9,.999)))
                group=sharedAdam.param_groups[-1]
            else:
                group=groups[name]
                for old in group['params']:
                    if old is not parameter:sharedAdam.state.pop(old,None)
                group['params']=[parameter];group['lr']=rates[name]
                group['eps']=1e-15;group['betas']=(.9,.999)
            saved=self._saved_groups.get(name)
            if saved is not None:
                group.update(_clone_tree(saved['hyper']))
                state=saved['state']
                sharedAdam.state.pop(parameter,None)
                if state:
                    sharedAdam.state[parameter]={k:(v.detach().cpu().clone() if k=='step' and torch.is_tensor(v) else
                        v.detach().to(parameter.device).clone() if torch.is_tensor(v) else v) for k,v in state.items()}
        self._saved_groups={}
        return self

    def training_setup(self,args):
        raise RuntimeError('Early replacements must attach to the existing shared Adam')

    def capture(self,sharedAdam=None):
        optimizer=self.optimizer if sharedAdam is None else sharedAdam
        states=_clone_tree(self._saved_groups,'cpu')
        if optimizer is not None:
            groups={g.get('name'):g for g in optimizer.param_groups}
            for name,attr in self.GROUPS:
                group=groups.get(name)
                if group is not None:
                    states[name]=dict(hyper={k:_clone_tree(v,'cpu') for k,v in group.items() if k!='params'},
                                     state=_clone_tree(optimizer.state.get(getattr(self,attr),{}),'cpu'))
        return dict(early_replacement=_clone_tree(dict(sh_degree=self.max_sh_degree,seed_raw=self._seed_raw,
            seed_alpha=self._seed_alpha,parent_rows=self.parent_rows,source_rows=self.source_rows,
            parameters={attr:getattr(self,attr) for _,attr in self.GROUPS},source_xyz=self._source_xyz,
            parameter_requires_grad={attr:getattr(self,attr).requires_grad for _,attr in self.GROUPS},
            source_q=self._source_q,codec_alpha=self.codec_alpha,spatial_lr_scale=self.spatial_lr_scale,
            optimizer_groups=states),'cpu'))

    def restore(self,packet,sharedAdam=None,args=None,device=None):
        state=packet['early_replacement'];device=device or state['seed_raw']['xyz'].device
        self.install(_clone_tree(state['seed_raw'],device),state['parent_rows'],_clone_tree(state['seed_alpha'],device))
        for _,attr in self.GROUPS:getattr(self,attr).data.copy_(state['parameters'][attr].to(device))
        for attr,enabled in state.get('parameter_requires_grad',{}).items():getattr(self,attr).requires_grad_(enabled)
        self._source_xyz=_clone_tree(state['source_xyz'],device);self._source_q=_clone_tree(state['source_q'],device)
        self.source_rows=_clone_tree(state['source_rows'],device)
        self.codec_alpha=bool(state['codec_alpha']);self.spatial_lr_scale=state['spatial_lr_scale']
        self._saved_groups=_clone_tree(state['optimizer_groups'],'cpu')
        if sharedAdam is not None:
            if args is None:raise ValueError('Shared Adam attachment needs the existing early learning rates')
            self.attach_optimizer(sharedAdam,args)
        return self

    def _indices(self,rows):
        value=torch.as_tensor(rows,device=self._xyz.device)
        if value.dtype==torch.bool and value.shape!=(len(self.parent_rows),):
            raise ValueError('Early subset mask must address the current learner layout')
        ids=value.nonzero().flatten() if value.dtype==torch.bool else value.long().flatten()
        if len(ids.unique())!=len(ids) or bool(((ids<0)|(ids>=len(self.parent_rows))).any()):
            raise ValueError('Early subset indices must be valid and unique')
        return ids

    def subset(self,rows):
        ids=self._indices(rows);state=self.capture()['early_replacement']
        state['seed_raw']={k:v[ids.cpu()] for k,v in state['seed_raw'].items()}
        for key in ('seed_alpha','parent_rows','source_rows','source_xyz','source_q'):
            if state[key] is not None:state[key]=state[key][ids.cpu()]
        for attr,value in list(state['parameters'].items()):
            state['parameters'][attr]=value[ids.cpu()]
            name=next(k for k,a in self.GROUPS if a==attr)
            for key,moment in list(state['optimizer_groups'].get(name,{}).get('state',{}).items()):
                if torch.is_tensor(moment) and moment.shape==value.shape:
                    state['optimizer_groups'][name]['state'][key]=moment[ids.cpu()]
        return type(self)(self.max_sh_degree).restore(dict(early_replacement=state),device=self._xyz.device)

    def select_rows_inplace(self,rows,sharedAdam=None):
        ids=self._indices(rows);optimizer=self.optimizer if sharedAdam is None else sharedAdam
        snapshot=self.subset(ids)
        for name,attr in self.GROUPS:
            old=getattr(self,attr);new=getattr(snapshot,attr)
            if optimizer is not None:
                for group in optimizer.param_groups:
                    if group.get('name')==name:group['params']=[new]
                state=optimizer.state.pop(old,{})
                if state:optimizer.state[new]={k:_clone_tree(v[ids] if torch.is_tensor(v) and v.shape==old.shape else v) for k,v in state.items()}
            setattr(self,attr,new)
        for key in ('_seed_raw','_seed_alpha','parent_rows','source_rows','_source_xyz','_source_q',
                    '_xyz','_rotation','_low','_high','_parent_max','max_radii2D','xyz_gradient_accum','denom'):
            setattr(self,key,getattr(snapshot,key))
        self._saved_groups={} if optimizer is not None else snapshot._saved_groups
        self.optimizer=optimizer
        return self

    subset_inplace=select_rows_inplace
