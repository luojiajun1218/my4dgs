"""Small CPU scene shared by adaptive-birth and frame-plan tests."""
from pathlib import Path
from types import MethodType, SimpleNamespace as NS
from unittest.mock import Mock
import torch
from torch import nn

from pipeline.early_replacement import EarlyReplacementController
from pipeline.model.birth_quantization import RAW_ATTRIBUTES, RAW_FIELDS
from pipeline.model.early_replacement_model import EarlyReplacementModel
from pipeline.model.gaussian import BirthGaussianModel
from pipeline.model.rigid_residual_tree import RigidTreeConfig
from pipeline.model.sparse_updates import OpacityUpdateConfig, QuantizedOpacityUpdate
from pipeline.model.spatial_gated_motion import SpatialGatedMotion
from pipeline.module import BirthConfig, Module, ModuleConfig


def early_model_fixture(dtype=torch.float64):
    raw = dict(xyz=torch.zeros(7, 3, dtype=dtype), rotation=torch.zeros(7, 4, dtype=dtype),
        logscale=torch.zeros(7, 3, dtype=dtype), opacity=torch.zeros(7, 1, dtype=dtype),
        sh_dc=torch.zeros(7, 1, 3, dtype=dtype), sh_rest=torch.zeros(7, 3, 3, dtype=dtype))
    raw['xyz'][:] = torch.arange(21, dtype=dtype).reshape(7, 3) * .031
    raw['rotation'][:] = torch.tensor([.93, .12, -.19, .27], dtype=dtype)
    raw['logscale'][:] = torch.tensor([-.4, -.9, -1.3], dtype=dtype)
    raw['opacity'][:] = torch.linspace(-.7, 1.3, 7, dtype=dtype)[:, None]
    raw['sh_dc'][:] = torch.linspace(-.2, .6, 21, dtype=dtype).reshape(7, 1, 3)
    raw['sh_rest'][:] = torch.linspace(-.07, .09, 63, dtype=dtype).reshape(7, 3, 3)
    rows = torch.tensor([4, 1])
    model = EarlyReplacementModel(1).install({k: v[rows].clone() for k, v in raw.items()}, rows)
    return model, raw


def optimizer(model):
    motion = nn.Parameter(torch.tensor([.13], dtype=model.local_xyz.dtype))
    shared = torch.optim.Adam([dict(params=[motion], name='motion', lr=.003)], eps=1e-8)
    args = NS(position_lr_init=.001, rotation_lr=.001, scaling_lr=.005, opacity_lr=.02, feature_lr=.01)
    model.attach_optimizer(shared, args)
    return shared, motion


class EvidenceRenderer:
    """CPU differentiable compositor; native renderer has separate checks."""
    collect_contributions = False
    def __init__(self, model):
        self.gaussians = model

    def __call__(self, view, bg, override_color=None):
        g = self.gaussians
        xyz, scale = g.get_xyz, g.get_scaling
        y, x = torch.meshgrid(torch.linspace(-.15, .65, 8), torch.linspace(-.15, .65, 8), indexing='ij')
        d = ((x[None]-xyz[:,0,None,None])/scale[:,0,None,None]).square() + ((y[None]-xyz[:,1,None,None])/scale[:,1,None,None]).square()
        alpha = g.get_opacity[:,0,None,None] * torch.exp(-.5*d)
        trans = torch.cumprod(torch.cat((torch.ones_like(alpha[:1]), 1-alpha), 0), 0)[:-1]
        colour = .5 + .28209479177387814*g.get_features[:,0] if override_color is None else override_color
        render = torch.einsum('nhw,nc->chw', alpha*trans, colour)
        n = len(xyz)
        mask = (alpha*trans).flatten(1).sum(1) > 0
        return dict(render=render, radii=torch.ones(n), visibility_filter=torch.ones(n,dtype=torch.bool), motion_contribution_filter=mask)


def early_controller_fixture(directory):
    xyz = torch.cartesian_prod(torch.tensor([.1,.3,.6,.9]), torch.tensor([.1,.9]), torch.tensor([2.9,3.1])).repeat_interleave(2, 0)
    n = len(xyz)
    old = BirthGaussianModel(1)
    ref = torch.linspace(-1.3579, 1.2345, n)[:, None]
    raw = dict(xyz=xyz, rotation=torch.tensor([[1.,0.,0.,0.]]).repeat(n,1), logscale=torch.full((n,3),-2.),
               opacity=ref, sh_dc=torch.full((n,1,3),.15), sh_rest=torch.zeros(n,3,3))
    for name, attr in zip(RAW_FIELDS, RAW_ATTRIBUTES):
        setattr(old, attr, nn.Parameter(raw[name].clone()))
    old.active_sh_degree = 1
    old.spatial_lr_scale = 1.
    old.max_radii2D = torch.zeros(n)
    old.xyz_gradient_accum = torch.zeros(n, 1)
    old.denom = torch.zeros(n, 1)
    names = dict(xyz='_xyz', f_dc='_features_dc', f_rest='_features_rest', opacity='_opacity', scaling='_scaling', rotation='_rotation')
    old.optimizer = torch.optim.Adam([dict(name=k, params=[getattr(old,v)]) for k,v in names.items()], lr=.001)
    cfg = ModuleConfig(birth=BirthConfig(mechanism='residual_transaction', radiance_mode='adaptive_residual_birth'))
    cfg.birth.residual.repair_candidates = 2
    cfg.birth.residual.max_candidates = 4
    cfg.rigid_tree.birth_steps = 100
    cfg.rigid_tree.rate_loss_weight = 0.
    update = QuantizedOpacityUpdate(OpacityUpdateConfig(enabled=True,mode='alpha',step=.01))
    update.setup(ref)
    with torch.no_grad():
        update.delta[0]=2.; update.delta[1]=-2.; update.delta[4]=1.5
        old._opacity.copy_(update.canonical_logits())
    update.commit(torch.arange(n))
    motion = SpatialGatedMotion(RigidTreeConfig(motion_steps=200,birth_steps=100,rate_loss_weight=0.,relative_motion_loss_weight=0.))
    motion.setup(xyz, raw['rotation'], 1., frame_index=1)
    with torch.no_grad():
        old._features_dc[2]=.123456; old._features_dc[3]=.341927
        old._features_rest[2]=torch.tensor([[.003456,-.012345,.008912],[-.001291,.004341,.009821],[.006101,.003212,-.003731]])
        old._features_rest[3]=.219371
        update.delta.zero_(); update.delta[4]=.05; update.delta[2]=.02
        old._opacity.copy_(update.canonical_logits())
    update.committed = False
    update.delta.requires_grad_(True)
    update.config.learning_rate = .0001
    for value in motion.parameters(): value.requires_grad_(True)
    for attr in RAW_ATTRIBUTES: getattr(old, attr).requires_grad_(False)
    m = NS(config=cfg, device=torch.device('cpu'), current_frame=1, gaussians=old,
        births=BirthGaussianModel(1), opacity_update=update, motion=motion,
        datamodule=NS(cameras_extent=1.), birth_transaction=None, birth_terminal=False,
        birth_control={}, birth_appearance_gradient=None, motion_steps=200, total_steps=300,
        motion_committed=False, frame_finalized=False, frame_prepared=True, full_update_active=False,
        early_initialized=False, early_birth_updates=0, early_shared_steps=0,
        early_selected=None, early_pinned_rows=None, early_pinned_values=None, early_focus_weights={},
        output_dir=Path(directory), trainer=NS(global_step=0), finalize_frame=Mock(),
        birth_cohort=0, birth_cohort_start=70, birth_end_step=300)
    views = {f'cam{i:02d}.png': NS(path=f'cam{i:02d}.png', camera_center=torch.tensor([i*.1,0.,0.]),
             image=torch.full((3,8,8),.55)) for i in range(1,21)}
    m.transaction_camera_views = lambda: views
    m.renderer = EvidenceRenderer(old)
    m.background = torch.zeros(3)
    for name, value in vars(EarlyReplacementController).items():
        if name.startswith('_early_') and callable(value):
            setattr(m, name, MethodType(value, m))
    for name in ('_rate_byte_price', '_opacity_rate_price', 'make_birth_view', 'forward', 'transaction_training_view',
                 'post_training_step', '_commit_opacity_and_retire'):
        setattr(m, name, MethodType(getattr(Module, name), m))
    m._early_attach_opacity()
    m.motion.record_final_diagnostics = lambda: None
    return m, views
