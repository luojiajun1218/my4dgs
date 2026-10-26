"""Surface tests."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import importlib
import math
import struct
import tempfile
import unittest
from types import SimpleNamespace, SimpleNamespace as NS
from unittest.mock import patch

import numpy as np
import torch

from pipeline.model import (
    contribution_rasterizer as legacy,
    residual_birth as rb,
    surface_component_evidence as sce,
)
from pipeline.model.birth_quantization import (
    RAW_ATTRIBUTES,
    RAW_FIELDS,
    materialize_quantized_gaussians,
    model_raw,
    patch_old_state,
)
from pipeline.model.gaussian import BirthGaussianModel
from pipeline.model.legacy_opacity_codec import decode_opacity_delta
from pipeline.model.rigid_residual_tree import RigidTreeConfig
from pipeline.model.rigid_tree_birth_codec import (
    ATTRIBUTES,
    decode_birth_attributes,
    encode_birth_attributes,
    encode_birth_symbols,
)
from pipeline.model.sparse_updates import (
    OpacityUpdateConfig,
    QuantizedOpacityUpdate,
    canonical_alpha_logits,
    encode_opacity_delta,
)
from pipeline.model.spatial_gated_motion import SpatialGatedMotion
from pipeline.model.surface_patch_model import C0, SurfacePatchModel
from pipeline.model.surface_patch_proposal import build_surface_patches
from pipeline.model.tracked_surface_proposal import (
    NAMES,
    _project,
    _residual_score,
    _select,
    _tracks,
    _triangulate,
    colmap_half_pixel_coordinates,
    propose_tracked_surface_births,
)

# ----- test_surface_patch_controller -----
MODULE=importlib.import_module('pipeline.module')


def fixture():
    xyz=torch.cartesian_prod(torch.tensor([.1,.3,.6,.9]),torch.tensor([.1,.9]),torch.tensor([2.9,3.1])).repeat_interleave(2,0)
    n=len(xyz);old=BirthGaussianModel(1)
    ref=torch.linspace(-1.3579,1.2345,n)[:,None]
    raw=dict(xyz=xyz,rotation=torch.tensor([[1.,0.,0.,0.]]).repeat(n,1),logscale=torch.full((n,3),-2.),
        opacity=ref,sh_dc=torch.full((n,1,3),.15),sh_rest=torch.zeros(n,3,3))
    for name,attr in zip(RAW_FIELDS,RAW_ATTRIBUTES):setattr(old,attr,torch.nn.Parameter(raw[name].clone()))
    old.active_sh_degree=1;old.spatial_lr_scale=1.;old.max_radii2D=torch.zeros(n)
    old.xyz_gradient_accum=torch.zeros(n,1);old.denom=torch.zeros(n,1)
    names=dict(xyz='_xyz',f_dc='_features_dc',f_rest='_features_rest',opacity='_opacity',scaling='_scaling',rotation='_rotation')
    old.optimizer=torch.optim.Adam([dict(name=k,params=[getattr(old,v)]) for k,v in names.items()],lr=.001)
    cfg=MODULE.ModuleConfig(birth=MODULE.BirthConfig(mechanism='residual_transaction',radiance_mode='surface_patch'))
    update=QuantizedOpacityUpdate(OpacityUpdateConfig(enabled=True,mode='alpha',step=.01))
    update.setup(ref);update.delta.data[0]=2.;update.delta.data[1]=-2.;update.delta.data[4]=1.5
    old._opacity.data.copy_(update.canonical_logits());update.commit(torch.arange(n))
    motion=SpatialGatedMotion(RigidTreeConfig(motion_steps=200,birth_steps=600,rate_loss_weight=0.,relative_motion_loss_weight=0.))
    motion.setup(xyz,raw['rotation'],1.,frame_index=1)
    views={f'fit{i}':NS(camera_center=torch.tensor([float(i)*.1,0.,0.])) for i in range(3)}
    m=NS(config=cfg,device=torch.device('cpu'),current_frame=1,gaussians=old,opacity_update=update,motion=motion,
         datamodule=NS(cameras_extent=1.),birth_terminal=False,birth_control={},birth_appearance_gradient=None,
         transaction_camera_views=lambda:views)
    m._set_patch_stage=lambda:MODULE.Module._set_patch_stage(m)
    m._surface_old_reference=lambda:MODULE.Module._surface_old_reference(m)
    m.make_birth_view=lambda reference=None:MODULE.Module.make_birth_view(m,reference)
    specs=[]
    for p,owned in enumerate(([0,1],[2])):
        specs.append(dict(center=torch.tensor([.3+p*.2,.4,3.]),frame=torch.eye(3),uv_bounds=torch.tensor([[-.2,-.2],[.2,.2]]),
          thickness=.01,kernel_uv=torch.tensor([[-.05,0.],[.05,0.]]),kernel_scale=torch.tensor([[.05,.04],[.05,.04]]),
          kernel_dc=torch.full((2,1,3),.33),kernel_sh_rest=torch.zeros(2,3,3),prior_dc=torch.full((2,1,3),.33),
          prior_sh_rest=torch.zeros(2,3,3),old_rows=torch.tensor(owned),source_rows=torch.tensor([owned[0]]*2),retention=.5,
          support_uv=torch.tensor([[-.1,-.1],[.1,-.1],[-.1,.1],[.1,.1]]),support_mass=torch.ones(4)))
    tx=rb.surface_transaction_from_specs(m,specs,tuple(views),tuple(views),{})
    MODULE.Module._install_birth_transaction(m,tx)
    return m,specs


class SurfaceControllerTests(unittest.TestCase):
    def test_selected_patch_old_opd_and_terminal_sources_roundtrip_exactly(self):
        m,_=fixture();before=m.gaussians._opacity.detach().clone();base=m.birth_transaction.old_base_symbols.clone()
        with torch.no_grad():m.births.patch_retention[0]=0
        mask=torch.tensor([True,True,False,False])
        trial=rb.encode_trial(m,mask,torch.zeros(len(before),dtype=torch.bool))
        torch.testing.assert_close(m.gaussians._opacity,before,atol=0,rtol=0)
        self.assertFalse(bool(torch.isin(trial.keep,torch.tensor([0,1])).any()))
        torch.testing.assert_close(trial.old_symbols[2:],base[2:],atol=0,rtol=0)
        packet=encode_opacity_delta(m.opacity_update.reference[trial.keep],trial.old_symbols[trial.keep],frame_index=1,step=.01,mode='alpha')
        decoded=decode_opacity_delta(packet,m.opacity_update.reference[trial.keep],frame_index=1)
        torch.testing.assert_close(decoded.logits,trial.raw['opacity'][:len(trial.keep)],atol=0,rtol=0)
        tx=rb.selected_population(m,trial);state=tx.capture()
        m.birth_terminal=True
        MODULE.Module._install_birth_transaction(m,rb.BirthTransaction.restore(state,'cpu'))
        self.assertEqual(m.births.patch_retention_cap,1.)
        with torch.no_grad():m.births.patch_retention.fill_(1.)
        _,_,symbols=patch_old_state(m)
        torch.testing.assert_close(symbols[:2],trial.old_symbols[:2],atol=0,rtol=0)
        view=m.make_birth_view();full,_=materialize_quantized_gaussians(m._surface_old_reference()[0],view)
        for name in RAW_FIELDS:torch.testing.assert_close(full[name],trial.raw[name],atol=0,rtol=0)
        self.assertTrue(bool((view.source<len(trial.keep)).all()))
        # This is the real final old-delta commit; no earlier operation touched
        # the incumbent or its opacity update parameter.
        m._commit_opacity_and_retire=lambda **kwargs:MODULE.Module._commit_opacity_and_retire(m,**kwargs)
        m.motion.record_final_diagnostics=lambda:None
        normalized=rb.CommitTrial(trial.keep,torch.arange(2),trial.sources,trial.raw,trial.complete_bytes,
            trial.motion_bytes,trial.birth_bytes,trial.opacity_bytes,trial.old_symbols)
        MODULE.Module._commit_population_trial(m,normalized)
        torch.testing.assert_close(m.opacity_update.canonical_logits(),trial.raw['opacity'][:len(trial.keep)],atol=0,rtol=0)
        view=m.make_birth_view();appearance={name:view.raw_value(name).detach() for name in ATTRIBUTES}
        packet=encode_birth_symbols(appearance,len(trial.keep),symbols={name:view.symbols[name] for name in ATTRIBUTES},
            steps={name:view.steps[name] for name in ATTRIBUTES},predictor={name:view.predictor[name] for name in ATTRIBUTES},source_rows=view.source)
        if packet is None:packet=encode_birth_attributes(appearance,len(trial.keep))
        base_raw=model_raw(m.gaussians)
        born=decode_birth_attributes(packet,base_raw,trial.raw['xyz'][len(trial.keep):],trial.raw['rotation'][len(trial.keep):],view.source)
        for name in ATTRIBUTES:torch.testing.assert_close(born[name],trial.raw[name][len(trial.keep):],atol=0,rtol=0)

    def test_patch_phase_checkpoint_and_invalid_fine_scale_do_not_expand_normal(self):
        m,_=fixture();capture=m.births.capture();tx=m.birth_transaction.capture()
        snapshot=dict(model=capture,transaction=tx)
        MODULE.Module._restore_birth_pool(m,snapshot)
        self.assertEqual(m.births.patch_retention_cap,.5)
        self.assertEqual(len(m.births.optimizer.state),0)
        with self.assertRaises(ValueError):m.births.patch_subset(torch.tensor([0]))
        # If a physical minimum is reached, subdivision retains that kernel.
        with torch.no_grad():m.births.patch_logscale2d.copy_(m.births.patch_scale_bounds[m.births._point_patch,0].log())
        before=m.births.patch_thickness.clone()
        result,report=rb.refine_surface_transaction(m,m.config.birth.residual,{})
        self.assertIsNone(result)
        torch.testing.assert_close(m.births.patch_thickness,before,atol=0,rtol=0)

    def test_fine_layout_installs_on_the_same_plane_and_ownership(self):
        m,_=fixture()
        views={name:NS(camera_center=torch.zeros(3),world_view_transform=torch.eye(4),
                      fovx=1.,fovy=1.,image=torch.ones(3,16,16)*.7)
               for name in m.birth_transaction.fit_names}
        m.forward=lambda view:{'render':torch.ones(3,16,16)*.1}
        refined,report=rb.refine_surface_transaction(m,m.config.birth.residual,views)
        self.assertIsNotNone(refined)
        self.assertGreater(len(refined.sources),len(m.births._xyz))
        old_frame=m.births.patch_frame.clone();old_thickness=m.births.patch_thickness.clone()
        old_owned=m.births.owned_old_rows.clone()
        MODULE.Module._install_birth_transaction(m,refined)
        torch.testing.assert_close(m.births.patch_frame,old_frame,atol=0,rtol=0)
        torch.testing.assert_close(m.births.patch_thickness,old_thickness,atol=0,rtol=0)
        torch.testing.assert_close(m.births.owned_old_rows,old_owned,atol=0,rtol=0)
        self.assertLessEqual(len(m.births._xyz),m.config.birth.residual.max_candidates)


# ----- test_surface_component_evidence -----
def camera(width=5, height=5, tanx=1., tany=1., dtype=torch.float64):
    view = torch.eye(4, dtype=dtype)
    projection = torch.zeros(4, 4, dtype=dtype)
    projection[0, 0], projection[1, 1] = 1/tanx, 1/tany
    projection[2, 2], projection[2, 3], projection[3, 2] = 100/(100-.01), -1/(100-.01), 1.
    return view, projection.T, tanx, tany, width, height


class ConditionalGeometryTests(unittest.TestCase):
    def test_single_front_sheet_retains_within_component_thickness(self):
        xyz = torch.tensor([[0., 0., 3.]], dtype=torch.float64)
        covariance = torch.diag(torch.tensor([.04, .01, .0001], dtype=torch.float64))[None]
        stats = sce.component_projection_statistics(xyz, covariance, *camera(1, 1))
        result = sce.conditional_component_evidence(stats, torch.tensor([[0]], dtype=torch.int32),
                                                   torch.tensor([[.4]]), torch.tensor([[.6]]))
        self.assertAlmostEqual(float(result['depth'][0, 0]), 3.)
        self.assertAlmostEqual(float(result['depth_variance'][0, 0]), .0001, places=10)
        self.assertTrue(bool(result['normal_unique'][0, 0]))
        self.assertLess(float(result['normal'][2, 0, 0]), 0.)

    def test_two_depth_layers_do_not_turn_association_spread_into_sheet_thickness(self):
        xyz = torch.tensor([[0., 0., 3.], [0., 0., 5.]], dtype=torch.float64)
        covariance = torch.diag(torch.tensor([.04, .01, .0001], dtype=torch.float64))[None].repeat(2, 1, 1)
        stats = sce.component_projection_statistics(xyz, covariance, *camera(1, 1))
        result = sce.conditional_component_evidence(stats, torch.tensor([[0]], dtype=torch.int32),
            torch.tensor([[.4]], dtype=torch.float64), torch.tensor([[.3]], dtype=torch.float64))
        mixture_weight = torch.tensor([.4, .3], dtype=torch.float64)/.7
        mixture_mean = (mixture_weight*xyz[:, 2]).sum()
        mixture_variance = (mixture_weight*(xyz[:, 2]-mixture_mean).square()).sum()+.0001
        self.assertGreater(float(mixture_variance), .9)
        self.assertAlmostEqual(float(result['depth_variance'][0, 0]), .0001)
        self.assertAlmostEqual(float(result['dominant_share'][0, 0]), 4/7)
        self.assertAlmostEqual(float(result['other_component_weight'][0, 0]), .3)

    def test_conditional_pixel_offset_sign_is_pixel_minus_mean(self):
        xyz = torch.tensor([[0., 0., 3.]], dtype=torch.float64)
        covariance = torch.tensor([[[.04, 0., .02], [0., .01, 0.], [.02, 0., .03]]], dtype=torch.float64)
        stats = sce.component_projection_statistics(xyz, covariance, *camera())
        ids = torch.zeros(5, 5, dtype=torch.int32)
        result = sce.conditional_component_evidence(stats, ids, torch.full((5, 5), .4), torch.full((5, 5), .6))
        self.assertGreater(float(result['depth'][2, 3]), 3.)
        self.assertLess(float(result['depth'][2, 1]), 3.)
        expected = .03-(stats['depth_gain'][0] *
            (stats['camera_covariance'][0, 2]@stats['projection_jacobian'][0].T)).sum()
        torch.testing.assert_close(stats['depth_variance'][0], expected)

    def test_native_aa_filter_and_fov_clamp_are_part_of_the_conditioning(self):
        xyz = torch.tensor([[10., 0., 1.]], dtype=torch.float64)
        covariance = torch.eye(3, dtype=torch.float64)[None]*.01
        stats = sce.component_projection_statistics(xyz, covariance, *camera(width=10, height=8))
        self.assertTrue(bool(stats['projection_clamped'][0]))
        self.assertAlmostEqual(float(stats['projection_jacobian'][0, 0, 2]), -5*1.3)
        projected = stats['projection_jacobian']@covariance@stats['projection_jacobian'].transpose(-1, -2)
        torch.testing.assert_close(stats['screen_covariance']-projected, torch.eye(2, dtype=torch.float64)[None]*.3)
        self.assertGreater(float(stats['depth_variance'][0]), 0.)

    def test_camera_rotation_uses_row_vector_covariance_convention(self):
        xyz = torch.tensor([[0., 0., 3.]], dtype=torch.float64)
        covariance = torch.diag(torch.tensor([.01, .04, .09], dtype=torch.float64))[None]
        view, projection, tx, ty, width, height = camera()
        theta = .2
        column_rotation = torch.tensor([[torch.cos(torch.tensor(theta)), 0., torch.sin(torch.tensor(theta))],
            [0., 1., 0.], [-torch.sin(torch.tensor(theta)), 0., torch.cos(torch.tensor(theta))]], dtype=torch.float64)
        view[:3, :3] = column_rotation.T
        stats = sce.component_projection_statistics(xyz, covariance, view, view@projection, tx, ty, width, height)
        torch.testing.assert_close(stats['camera_covariance'], column_rotation[None]@covariance@column_rotation.T[None])

    def test_normal_ambiguity_and_invalid_covariance_are_not_false_certainty(self):
        xyz = torch.tensor([[0., 0., 3.], [0., 0., 3.], [0., 0., 0.]], dtype=torch.float64)
        covariance = torch.eye(3, dtype=torch.float64)[None].repeat(3, 1, 1)*.01
        covariance[1, 0, 0] = -.1
        stats = sce.component_projection_statistics(xyz, covariance, *camera())
        self.assertEqual(stats['valid'].tolist(), [True, False, False])
        self.assertFalse(bool(stats['normal_unique'][0]))
        self.assertEqual(float(stats['normal_anisotropy'][0]), 0.)
        self.assertTrue(bool(torch.isnan(stats['depth_variance'][1:] ).all()))

    def test_missing_component_returns_invalid_geometry_not_zero_variance(self):
        stats = sce.component_projection_statistics(torch.empty(0, 3, dtype=torch.float64),
                                                     torch.empty(0, 3, 3, dtype=torch.float64), *camera(2, 1))
        result = sce.conditional_component_evidence(stats, torch.full((1, 2), -1, dtype=torch.int32),
            torch.zeros(1, 2), torch.ones(1, 2))
        self.assertFalse(bool(result['component_valid'].any()))
        self.assertTrue(bool(torch.isnan(result['depth_variance']).all()))
        self.assertEqual(result['dominant_share'].tolist(), [[0., 0.]])

    def test_nonpositive_linearized_ray_depth_is_not_valid_surface_geometry(self):
        xyz = torch.tensor([[0., 0., 3.]], dtype=torch.float64)
        covariance = torch.eye(3, dtype=torch.float64)[None]
        stats = sce.component_projection_statistics(xyz, covariance, *camera())
        # A very broad tilted component may extrapolate through the camera;
        # it must not produce a valid negative-depth surface observation.
        stats['depth_gain'][0] = torch.tensor([5., 0.])
        result = sce.conditional_component_evidence(stats, torch.zeros(5, 5, dtype=torch.int32),
                                                     torch.full((5, 5), .4), torch.full((5, 5), .6))
        self.assertFalse(bool(result['component_valid'][2, 0]))
        self.assertTrue(bool(torch.isnan(result['depth_variance'][2, 0])))

    def test_packed_covariance_matches_full_matrix_and_native_supplied_rotation(self):
        scales = torch.tensor([[.2, .1, .01]], dtype=torch.float64)
        q = torch.tensor([[.8, .1, .2, -.3]], dtype=torch.float64)
        q = q/q.norm(dim=-1, keepdim=True)
        full = sce.covariance_from_scale_rotation(scales, q)
        packed = full[:, (0, 0, 0, 1, 1, 2), (0, 1, 2, 1, 2, 2)]
        torch.testing.assert_close(sce.unpack_covariance(packed), full)
        eigenvalues = torch.linalg.eigvalsh(full)
        torch.testing.assert_close(eigenvalues, scales.square().sort(-1).values)


class CompatibilityTests(unittest.TestCase):
    def test_legacy_three_outputs_observed_semantics_and_backward_arity_stay_unchanged(self):
        count = 2
        xyz = torch.zeros(count, 3, requires_grad=True)
        screen = torch.zeros(count, 3, requires_grad=True)
        color = torch.full((count, 3), .5, requires_grad=True)
        alpha = torch.full((count, 1), .3, requires_grad=True)
        scale = torch.ones(count, 3, requires_grad=True)
        q = torch.tensor([[1., 0., 0., 0.]]).repeat(count, 1).requires_grad_()
        settings = legacy.GaussianRasterizationSettings(1, 1, 1., 1., torch.zeros(3),
            1., torch.eye(4), torch.eye(4), 0, torch.zeros(3), False, False)
        class Backend:
            def rasterize_gaussians(self, *args):
                return (2, torch.full((3, 1, 1), .5), torch.tensor([2, 0], dtype=torch.int32),
                        torch.empty(0), torch.empty(0), torch.empty(0), torch.tensor([1, 0], dtype=torch.int32))
            def rasterize_gaussians_backward(self, *args):
                return (torch.ones_like(screen), torch.ones_like(color), torch.ones_like(alpha),
                        torch.ones_like(xyz), torch.empty(0), torch.empty(0), torch.ones_like(scale), torch.ones_like(q))
        with patch.object(legacy, '_backend', return_value=Backend()):
            output = legacy.GaussianRasterizer(settings)(xyz, screen, alpha, colors_precomp=color, scales=scale, rotations=q)
            self.assertEqual(len(output), 3)
            self.assertEqual(output[2].tolist(), [1, 0])
            output[0].sum().backward()
        for parameter in (xyz, screen, color, alpha, scale, q):
            torch.testing.assert_close(parameter.grad, torch.ones_like(parameter))

    def test_aux_cpu_inputs_fail_before_any_backend_compile_or_renderer(self):
        with patch.object(sce, '_surface_backend', side_effect=AssertionError('unexpected build')):
            with self.assertRaisesRegex(ValueError, 'CUDA'):
                sce.SurfaceComponentRasterizer(NS())(torch.zeros(1, 3))

    def test_independent_bridge_does_not_mutate_existing_renderer_state(self):
        marker, diagnostics = object(), object()
        gaussians = NS(get_xyz=torch.zeros(1, 3), get_opacity=torch.ones(1, 1)*.5,
            get_scaling=torch.ones(1, 3), get_rotation=torch.tensor([[1., 0., 0., 0.]]), active_sh_degree=0)
        renderer = NS(gaussians=gaussians, collect_contributions=True, last_contributions=diagnostics,
            config=NS(debug=False, compute_cov3D_python=False),
            apply_color=lambda kwargs, view: dict(kwargs, colors_precomp=torch.ones(1, 3)))
        view = NS(image=torch.zeros(3, 2, 2), fovx=1., fovy=1., world_view_transform=torch.eye(4),
                  full_proj_transform=torch.eye(4), camera_center=torch.zeros(3))
        class Rasterizer:
            def __init__(self, settings): self.settings = settings
            def __call__(self, **kwargs): return marker
        with patch.object(sce, 'SurfaceComponentRasterizer', Rasterizer):
            self.assertIs(sce.render_surface_components(renderer, view, torch.zeros(3)), marker)
        self.assertIs(renderer.gaussians, gaussians)
        self.assertTrue(renderer.collect_contributions)
        self.assertIs(renderer.last_contributions, diagnostics)


# ----- test_surface_patch_proposal -----
def scene(two_planes=False,flat=False):
    xs=torch.linspace(-.8,.8,12);ys=torch.linspace(-.6,.6,9)
    x,y=torch.meshgrid(xs,ys,indexing='ij');x,y=x.flatten(),y.flatten()
    if two_planes:
        keep=x.abs()>.15;x,y=x[keep],y[keep]
    z=torch.where(x>0,4.,3.) if two_planes else torch.full_like(x,3.)
    xyz=torch.stack((x,y,z),-1);n=len(xyz)
    base=dict(xyz=xyz,rotation=torch.tensor([[1.,0.,0.,0.]]).repeat(n,1),
        logscale=torch.tensor([[.08,.06,.002]]).log().repeat(n,1),
        opacity=torch.full((n,1),torch.logit(torch.tensor(.8))),
        sh_dc=torch.zeros(n,1,3),sh_rest=torch.full((n,3,3),.01))
    evidence=[];width,height,focal=96,80,80.
    px,py=torch.meshgrid(torch.arange(width),torch.arange(height),indexing='xy')
    for i,camera_x in enumerate((-.1,0.,.1)):
        depth=torch.where(px<width/2,3.,4.) if two_planes else torch.full((height,width),3.)
        world_x=(px+.5-width/2)/focal*depth+camera_x
        world_y=(py+.5-height/2)/focal*depth
        color=torch.stack((.4+.08*torch.sin(3*world_x),
                           .5+.08*torch.cos(3*world_y),
                           .3+.05*torch.sin(world_x+world_y)))
        if flat:color.fill_(.4)
        matrix=torch.eye(4);matrix[3,0]=-camera_x
        view=SimpleNamespace(path=f'cam{i+1:02d}.png',image=color,
            fovx=2*math.atan(width/(2*focal)),fovy=2*math.atan(height/(2*focal)),
            world_view_transform=matrix)
        residual=torch.zeros_like(color);residual[:,16:64,16:80]=.08
        normal=torch.zeros_like(color);normal[2]=1
        evidence.append(dict(view=view,current_gt=color,previous_image=color.clone(),
            residual_rgb=residual,coverage=torch.ones(height,width),depth=depth,
            depth_variance=torch.zeros(height,width),normal=normal,
            observed_old=torch.ones(n,dtype=torch.bool)))
    return base,evidence


class SurfacePatchProposalTests(unittest.TestCase):
    def test_partial_image_cells_preserve_error_mass_without_padding_extrapolation(self):
        from pipeline.model.surface_patch_proposal import _pooled_sum
        image=torch.ones(13,1345,dtype=torch.float64)*2
        pooled=_pooled_sum(image)
        torch.testing.assert_close(pooled.sum(),image.sum(),atol=0,rtol=0)
        self.assertEqual(float(pooled[-1,-1]),5*1*2)
        self.assertEqual(float(pooled[0,-1]),8*1*2)

    def test_persistent_DC_region_spans_many_sources_and_has_area(self):
        base,evidence=scene()
        specs,report=build_surface_patches(base,evidence,max_kernels=24,max_patches=4)
        self.assertGreater(len(specs),0)
        self.assertTrue(report['persistent_mean_residual_included'])
        self.assertEqual(report['temporal_evidence_views'],3)
        self.assertFalse(report['external_test_used'])
        spec=max(specs,key=lambda s:len(s['old_rows']))
        self.assertGreater(len(spec['old_rows']),20)
        # The chart is a residual surface domain, not the nearest GS's 1 sigma.
        self.assertGreater(float((spec['uv_bounds'][1]-spec['uv_bounds'][0]).max()),10*.08)
        self.assertGreater(float((spec['kernel_uv'].amax(0)-spec['kernel_uv'].amin(0)).max()),6*.08)
        self.assertGreater(spec['score'],0)
        self.assertLessEqual(sum(len(s['kernel_uv']) for s in specs),24)
        torch.testing.assert_close(spec['frame'].T@spec['frame'],torch.eye(3),atol=1e-6,rtol=0)
        self.assertGreater(float(torch.det(spec['frame'])),.99999)
        self.assertIsNone(spec['diagnostics']['actual_native_new_coverage'])
        self.assertIn('not_native_rendered',spec['diagnostics']['area_scope'])

    def test_flat_low_frequency_error_is_not_discarded_as_no_contrast(self):
        base,evidence=scene(flat=True)
        specs,_=build_surface_patches(base,evidence,12,2)
        self.assertGreater(len(specs),0)
        for spec in specs:
            torch.testing.assert_close(spec['kernel_color'],torch.full_like(spec['kernel_color'],.4),atol=1e-6,rtol=0)
            self.assertGreater(spec['diagnostics']['assigned_error_mass'],0)

    def test_depth_discontinuity_produces_separate_unique_ownership(self):
        base,evidence=scene(two_planes=True)
        specs,_=build_surface_patches(base,evidence,24,4)
        self.assertGreaterEqual(len(specs),2)
        all_rows=torch.cat([s['old_rows'] for s in specs])
        self.assertEqual(len(all_rows.unique()),len(all_rows))
        for spec in specs:
            source_depth=base['xyz'][spec['old_rows'],2]
            self.assertLess(float(source_depth.max()-source_depth.min()),1e-6)
            self.assertLess(spec['diagnostics']['plane_error_max'],1e-5)

    def test_crossview_depth_disagreement_cannot_certify_an_old_surface(self):
        base,evidence=scene()
        evidence[1]['depth'].fill_(5.)
        evidence[2]['depth'].fill_(6.)
        specs,report=build_surface_patches(base,evidence,12,2)
        self.assertEqual(specs,[])
        self.assertGreater(report['rejected_crossview_surface'],0)

    def test_conflicting_normals_and_transparent_depth_spread_are_rejected(self):
        base,evidence=scene()
        for e in evidence[1:]:
            e['normal'].zero_();e['normal'][0]=1
        specs,_=build_surface_patches(base,evidence,12,2)
        self.assertEqual(specs,[])
        base,evidence=scene()
        for e in evidence:e['depth_variance'].fill_(.5)
        specs,_=build_surface_patches(base,evidence,12,2)
        self.assertEqual(specs,[])

    def test_geometric_plane_alone_does_not_approve_inconsistent_RGB(self):
        base,evidence=scene()
        generator=torch.Generator().manual_seed(19)
        for e in evidence[1:]:
            color=torch.rand(e['current_gt'].shape,generator=generator)
            e['current_gt']=color;e['previous_image']=color.clone()
        specs,report=build_surface_patches(base,evidence,12,2)
        self.assertEqual(specs,[])
        self.assertGreater(report['rejected_crossview_surface'],0)

    def test_native_observation_is_required_and_unobserved_rows_cannot_own(self):
        base,evidence=scene()
        for e in evidence:e['observed_old'].zero_()
        specs,report=build_surface_patches(base,evidence,12,2)
        self.assertEqual(specs,[])
        self.assertGreater(report['rejected_without_multiple_old'],0)
        del evidence[0]['observed_old']
        with self.assertRaisesRegex(ValueError,'native old contribution'):
            build_surface_patches(base,evidence,12,2)

    def test_test_camera_and_insufficient_views_are_not_accepted(self):
        base,evidence=scene()
        evidence[0]['view'].path='cam00.png'
        with self.assertRaisesRegex(ValueError,'training cameras'):
            build_surface_patches(base,evidence,12,2)
        base,evidence=scene()
        with self.assertRaisesRegex(ValueError,'three training views'):
            build_surface_patches(base,evidence[:2],12,2)

    def test_material_SH_provenance_is_distinct_from_codec_source_and_immutable(self):
        base,evidence=scene()
        specs,_=build_surface_patches(base,evidence,12,2)
        self.assertGreater(len(specs),0)
        for spec in specs:
            self.assertTrue(bool(((spec['kernel_color']>=0)&(spec['kernel_color']<=1)).all()))
            self.assertTrue(bool(torch.isin(spec['material_rows'],spec['old_rows']).all()))
            self.assertTrue(bool(torch.isin(spec['source_rows'],spec['old_rows']).all()))
            torch.testing.assert_close(spec['material_weights'].sum(-1),torch.ones(len(spec['kernel_uv'])),atol=1e-6,rtol=0)
            torch.testing.assert_close(spec['kernel_sh_rest'],torch.full_like(spec['kernel_sh_rest'],.01),atol=1e-7,rtol=0)
            torch.testing.assert_close(spec['prior_sh_rest'],spec['kernel_sh_rest'])
            self.assertEqual(len(spec['support_uv']),len(spec['support_color']))
            self.assertEqual(len(spec['support_uv']),len(spec['support_mass']))
        before=specs[0]['prior_sh_rest'].clone()
        base['sh_rest'].fill_(300.)
        torch.testing.assert_close(specs[0]['prior_sh_rest'],before,atol=0,rtol=0)

    def test_DC_subtracts_inherited_angular_SH_instead_of_adding_it_twice(self):
        from pipeline.model import surface_patch_proposal as proposal
        base,evidence=scene(flat=True)
        C1=.4886025119029199
        amplitude=torch.tensor([2.,1.5,1.])
        dc=(torch.tensor([.4,.4,.4])-.5-C1*amplitude)/proposal._C0
        base['sh_rest'].zero_();base['sh_rest'][:,1]=amplitude
        base['sh_dc'][:,0]=dc
        for e in evidence:
            h,w=e['depth'].shape
            y,x=torch.meshgrid(torch.arange(h),torch.arange(w),indexing='ij')
            ray_z=1/torch.sqrt(((x+.5-w/2)/80.)**2+((y+.5-h/2)/80.)**2+1)
            image=.5+proposal._C0*dc[:,None,None]+C1*amplitude[:,None,None]*ray_z
            self.assertGreater(float(image.min()),0)
            self.assertLess(float(image.max()),1)
            e['current_gt']=image;e['previous_image']=image.clone()
        specs,_=build_surface_patches(base,evidence,12,2)
        self.assertGreater(len(specs),0)
        for spec in specs:
            torch.testing.assert_close(spec['kernel_dc'][:,0],dc.expand(len(spec['kernel_uv']),3),atol=3e-4,rtol=0)
            torch.testing.assert_close(spec['prior_dc'],spec['kernel_dc'],atol=0,rtol=0)
            xyz=spec['center']+spec['kernel_uv']@spec['frame'][:,:2].T
            for e in evidence:
                cache=proposal._cache_evidence(e,len(base['xyz']),base['xyz'].dtype,0)
                valid,target=proposal._visible_support(cache,xyz,spec['normal'])
                direction=torch.nn.functional.normalize(xyz-cache['inverse'][3,:3],dim=-1)
                actual=.5+proposal._C0*spec['kernel_dc'][:,0]+C1*amplitude*direction[:,2:3]
                torch.testing.assert_close(actual[valid],target[valid],atol=9e-5,rtol=0)
                self.assertGreater(float(actual.min()),0)
                self.assertLess(float(actual.max()),1)
            # Naive RGB2SH plus inherited rest would create an HDR curve here.
            naive=spec['kernel_color']+C1*amplitude
            self.assertGreater(float(naive.max()),1.)

    def test_unobserved_kernel_center_falls_back_to_physical_material_DC(self):
        from pipeline.model import surface_patch_proposal as proposal
        base,evidence=scene(flat=True)
        base['sh_dc'].fill_(-2.)
        specs,_=build_surface_patches(base,evidence,12,2)
        spec=specs[0];points=spec['support_world'];count=len(points)
        candidate=dict(center=spec['center'],frame=spec['frame'],xyz=points,
            weight=spec['support_mass'],color=spec['support_color'],
            footprint=torch.full((count,),.3),area=torch.ones(count),
            old_set=set(spec['old_rows'].tolist()),views=set(spec['diagnostics']['support_views']))
        caches=[proposal._cache_evidence(e,len(base['xyz']),base['xyz'].dtype,i) for i,e in enumerate(evidence)]
        def unseen(cache,xyz,normal):
            return torch.zeros(len(xyz),dtype=torch.bool),torch.full((len(xyz),3),.4)
        with patch.object(proposal,'_visible_support',side_effect=unseen):
            result=proposal._materialize(candidate,1,base,torch.zeros_like(base['xyz']),caches)
        torch.testing.assert_close(result['kernel_dc'],torch.full_like(result['kernel_dc'],-2.),atol=1e-6,rtol=0)


# ----- test_surface_patch_model -----
def rotation(q):
    q = q/q.norm(dim=-1, keepdim=True)
    w, x, y, z = q.unbind(-1)
    return torch.stack((1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
        2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
        2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)), -1).reshape(-1, 3, 3)


def specs(dtype=torch.float64):
    angle = .37
    frame = torch.tensor([[math.cos(angle), 0., math.sin(angle)],
                          [0., 1., 0.], [-math.sin(angle), 0., math.cos(angle)]], dtype=dtype)
    return [dict(center=torch.tensor([0., 0., 3.], dtype=dtype), frame=frame,
        uv_bounds=torch.tensor([[-.3, -.2], [.4, .25]], dtype=dtype), thickness=.01,
        kernel_uv=torch.tensor([[-.12, -.04], [.04, .07], [.15, -.07]], dtype=dtype),
        kernel_scale=torch.tensor([[.04, .03], [.06, .025], [.03, .05]], dtype=dtype),
        kernel_color=torch.tensor([[.3, .4, .5], [.6, .5, .4], [.2, .3, .4]], dtype=dtype),
        old_rows=torch.tensor([0, 2]), source_rows=torch.tensor([0, 2, 0])),
        dict(center=torch.tensor([.5, 0., 3.2], dtype=dtype),
        frame=torch.diag(torch.tensor([1., -1., -1.], dtype=dtype)),
        uv_bounds=torch.tensor([[-.02, .03], [.11, .13]], dtype=dtype), thickness=.005,
        kernel_uv=torch.tensor([[0., .06], [.08, .09]], dtype=dtype),
        kernel_scale=torch.tensor([[.025, .02], [.03, .03]], dtype=dtype),
        kernel_color=torch.tensor([[.4, .5, .6], [.5, .4, .3]], dtype=dtype),
        kernel_angle=torch.tensor([.2, -.3], dtype=dtype),
        old_rows=torch.tensor([1, 4]), source_rows=torch.tensor([1, 4]))]


def arguments():
    return NS(percent_dense=.01, position_lr_init=.001, position_lr_final=.0001,
        position_lr_delay_mult=.01, position_lr_max_steps=100, feature_lr=.01,
        opacity_lr=.02, scaling_lr=.005, rotation_lr=.001)


def config():
    return NS(new_xyz_step_extent=.03, new_rotation_step=.017,
              new_logscale_step=.03, new_opacity_step=.02, new_sh_step=.006,
              rotation_updates=True)


def reference(dtype=torch.float64):
    return dict(xyz=torch.zeros(6, 3, dtype=dtype), rotation=torch.tensor([[1., 0., 0., 0.]], dtype=dtype).repeat(6, 1),
        logscale=torch.zeros(6, 3, dtype=dtype), opacity=torch.linspace(-.6, .6, 6, dtype=dtype)[:, None],
        sh_dc=torch.zeros(6, 1, 3, dtype=dtype), sh_rest=torch.zeros(6, 3, 3, dtype=dtype))


def loss(model):
    raw = model.patch_phenotype()
    weights = torch.arange(1, len(model._xyz)*3+1, dtype=model._xyz.dtype).reshape(-1, 3)
    return ((raw['xyz']*weights).sum()+raw['logscale'].sum()+
            (raw['rotation']*torch.tensor([.2, .3, .4, .5], dtype=model._xyz.dtype)).sum()+
            raw['sh_dc'].sum()+raw['opacity'].sigmoid().sum()+
            model.raw_old_alpha(reference(model._xyz.dtype)['opacity']).sum())


class SurfaceGeometryTests(unittest.TestCase):
    def test_precomputed_covariance_uses_derived_angle_and_fixed_plane_normal(self):
        model = SurfacePatchModel(1).install_patches(specs())
        values = model.get_covariance()
        matrix = torch.stack((values[:, 0], values[:, 1], values[:, 2], values[:, 1],
                              values[:, 3], values[:, 4], values[:, 2], values[:, 4], values[:, 5]), -1).reshape(-1, 3, 3)
        normal = model.patch_frame[model._point_patch, :, 2]
        variance = (normal*(matrix@normal[:, :, None])[:, :, 0]).sum(-1)
        torch.testing.assert_close(variance, model.patch_thickness[model._point_patch, 0].square(), rtol=1.e-12, atol=1.e-18)
        values.square().sum().backward()
        self.assertGreater(float(model.patch_logscale2d.grad.abs().sum()), 0.)

    def test_projection_and_quantized_pose_never_move_off_plane_or_tilt_normal(self):
        model = SurfacePatchModel(1).install_patches(specs(), old_count=6)
        with torch.no_grad():
            model.patch_uv[0] = torch.tensor([99., -99.])
            model.patch_logscale2d[1] = torch.tensor([100., -100.])
            model.patch_angle.add_(15.)
            model._features_dc.fill_(500.)
        model.project_parameters()
        raw = model.patch_quantized_raw(config(), 1., reference(), model.source_rows)
        p = model._point_patch
        normal = model.patch_frame[p, :, 2]
        residual = ((raw['xyz']-model.patch_center[p])*normal).sum(-1)
        torch.testing.assert_close(residual, torch.zeros_like(residual), rtol=0, atol=2.e-16)
        actual_normal = rotation(raw['rotation'])[:, :, 2]
        torch.testing.assert_close(actual_normal, normal, rtol=0, atol=5.e-16)
        torch.testing.assert_close(raw['logscale'][:, 2].exp(), model.patch_thickness[p, 0], rtol=0, atol=1.e-17)
        local = torch.einsum('nki,nk->ni', model.patch_frame[p], raw['xyz']-model.patch_center[p])[:, :2]
        self.assertTrue(bool(((local >= model.patch_uv_bounds[p, 0]-1.e-15) &
                              (local <= model.patch_uv_bounds[p, 1]+1.e-15)).all()))
        self.assertTrue(bool(((raw['sh_dc']*C0+.5 >= 0) & (raw['sh_dc']*C0+.5 <= 1)).all()))

    def test_all_joint_leaves_receive_real_gradients_and_frozen_geometry_does_not(self):
        model = SurfacePatchModel(1).install_patches(specs(), old_count=6)
        model.spatial_lr_scale = 2.
        model.training_setup(arguments())
        loss(model).backward()
        for parameter in (model.patch_uv, model.patch_logscale2d, model.patch_angle,
                          model._features_dc, model._opacity, model.patch_retention):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()))
            self.assertGreater(float(parameter.grad.abs().sum()), 0.)
        for parameter in (model._xyz, model._rotation, model._scaling, model._features_rest):
            self.assertFalse(parameter.requires_grad)
            self.assertIsNone(parameter.grad)
        # Default old retention is exactly 1, yet its endpoint gradient exists.
        torch.testing.assert_close(model.patch_retention, torch.ones_like(model.patch_retention))
        model.optimizer.step(); model.project_parameters()
        self.assertLess(float(model.patch_retention.max()), 1.)

    def test_quantized_local_pose_has_nonzero_ste_gradients_without_world_rounding(self):
        model = SurfacePatchModel(1).install_patches(specs(), old_count=6)
        raw = model.patch_quantized_raw(config(), 1., reference(), model.source_rows)
        weights = torch.arange(1, 16, dtype=torch.float64).reshape(5, 3)
        objective = ((raw['xyz']*weights).sum()+raw['logscale'].sum()+
            (raw['rotation']*torch.tensor([.3, .2, .4, .5])).sum()+
            raw['opacity'].sum()+raw['sh_dc'].sum())
        objective.backward()
        for parameter in (model.patch_uv, model.patch_logscale2d, model.patch_angle,
                          model._features_dc, model._opacity):
            self.assertGreater(float(parameter.grad.abs().sum()), 0.)

    def test_old_ownership_is_unique_and_nonowners_keep_exact_alpha(self):
        model = SurfacePatchModel(1).install_patches(specs(), old_count=6)
        raw_opacity = reference()['opacity']
        initial = raw_opacity.clone()
        model.patch_retention.data[:] = torch.tensor([[.25], [.6]])
        alpha = model.raw_old_alpha(raw_opacity)
        torch.testing.assert_close(alpha[[3, 5]], raw_opacity.sigmoid()[[3, 5]], rtol=0, atol=0)
        torch.testing.assert_close(alpha[[0, 2]], raw_opacity.sigmoid()[[0, 2]]*.25)
        torch.testing.assert_close(raw_opacity, initial, rtol=0, atol=0)
        alpha.sum().backward()
        self.assertGreater(float(model.patch_retention.grad.abs().sum()), 0.)
        bad = specs(); bad[1]['old_rows'] = torch.tensor([0, 4])
        with self.assertRaisesRegex(ValueError, 'ownership'):
            SurfacePatchModel(1).install_patches(bad)


class SurfaceRadianceTests(unittest.TestCase):
    def test_explicit_angular_aware_prior_initializes_dc_without_adding_sh_twice(self):
        supplied = specs()[:1]
        prior = torch.full((3, 1, 3), -.7, dtype=torch.float64)
        supplied[0]['prior_dc'] = prior
        supplied[0]['kernel_sh_rest'] = torch.full((3, 3, 3), .4, dtype=torch.float64)
        model = SurfacePatchModel(1).install_patches(supplied,
            views=torch.tensor([[0., 0., 0.]], dtype=torch.float64))
        torch.testing.assert_close(model._features_dc, prior, rtol=0, atol=0)
        torch.testing.assert_close(model.patch_prior_dc, prior, rtol=0, atol=0)

    def test_codec_hdr_source_does_not_expand_trusted_ldr_material_box(self):
        model = SurfacePatchModel(1).install_patches(specs())
        base = reference()
        base['sh_dc'][0].fill_(250.)
        base['sh_dc'][2].fill_(-200.)
        model._features_dc.data.fill_(1000.)
        raw = model.patch_quantized_raw(config(), 1., base, model.source_rows)
        rgb = raw['sh_dc']*C0+.5
        self.assertTrue(bool(((rgb >= 0) & (rgb <= 1)).all()))
        low, high = model.radiance_dc_bounds()
        torch.testing.assert_close(high, torch.full_like(high, .5/C0))
        self.assertTrue(bool((raw['sh_rest'] == 0).all()))
        # Switching arbitrary entropy predictors changes only feasible grid,
        # never the physical material envelope.
        after_low, after_high = model.radiance_dc_bounds()
        torch.testing.assert_close(low, after_low, rtol=0, atol=0)
        torch.testing.assert_close(high, after_high, rtol=0, atol=0)

    def test_trusted_hdr_and_directional_sh_remain_bounded_by_material_prior(self):
        supplied = specs()[:1]
        supplied[0]['kernel_dc'] = torch.tensor([[[3., -3., .1]]], dtype=torch.float64).repeat(3, 1, 1)
        rest = torch.zeros(3, 3, 3, dtype=torch.float64)
        rest[:, 2, 0] = .3
        supplied[0]['kernel_sh_rest'] = rest
        model = SurfacePatchModel(1).install_patches(supplied)
        with self.assertRaisesRegex(ValueError, 'radiance views'):
            model.patch_phenotype()
        model.set_radiance_views(torch.tensor([[-.2, 0., 0.], [.3, .1, 0.]], dtype=torch.float64))
        model._features_dc.data[0, 0] = torch.tensor([300., -300., 20.])
        model.project_parameters()
        low, high = model.radiance_dc_bounds()
        self.assertTrue(bool(((model._features_dc >= low) & (model._features_dc <= high)).all()))
        self.assertGreater(float(high[0, 0, 0]), .5/C0)
        self.assertLess(float(low[0, 0, 1]), -.5/C0)
        torch.testing.assert_close(model._features_rest, rest, rtol=0, atol=0)


class SurfaceLifecycleTests(unittest.TestCase):
    def test_retention_probation_cap_survives_capture_subset_and_terminal_release(self):
        model = self.trained().set_retention_cap(.5)
        model.patch_retention.data.fill_(.9)
        old = reference()['opacity']
        alpha = model.raw_old_alpha(old)
        torch.testing.assert_close(alpha[model.owned_old_rows], old.sigmoid()[model.owned_old_rows]*.5)
        model.project_parameters()
        restored = SurfacePatchModel(1).restore(model.capture(), arguments())
        selected = restored.patch_subset(torch.tensor([3, 4]))
        self.assertEqual(selected.patch_retention_cap, .5)
        torch.testing.assert_close(selected.patch_retention, torch.full_like(selected.patch_retention, .5))
        selected.set_retention_cap(1.)
        selected.patch_retention.data.fill_(.8)
        selected.project_parameters()
        self.assertEqual(selected.patch_retention_cap, 1.)
        self.assertAlmostEqual(float(selected.patch_retention[0, 0]), .8)

    def trained(self):
        model = SurfacePatchModel(1).install_patches(specs(), old_count=6)
        model.spatial_lr_scale = 2.
        model.training_setup(arguments())
        loss(model).backward(); model.optimizer.step(); model.optimizer.zero_grad(set_to_none=True)
        model.project_parameters()
        return model

    def test_capture_restore_preserves_all_latents_priors_and_adam_without_aliases(self):
        model = self.trained()
        packet = model.capture()
        restored = SurfacePatchModel(1).restore(packet, arguments())
        for name in RAW_FIELDS:
            torch.testing.assert_close(model_raw(restored)[name], model_raw(model)[name], rtol=0, atol=0)
        torch.testing.assert_close(restored.patch_retention, model.patch_retention, rtol=0, atol=0)
        for old, new in zip(model.optimizer.param_groups, restored.optimizer.param_groups):
            self.assertEqual(old['lr'], new['lr'])
            for key, value in model.optimizer.state[old['params'][0]].items():
                torch.testing.assert_close(restored.optimizer.state[new['params'][0]][key], value, rtol=0, atol=0)
        restored.patch_prior_dc[0].fill_(99.)
        self.assertFalse(bool((model.patch_prior_dc[0] == 99.).all()))
        self.assertFalse(bool((packet['patch']['prior_dc'][0] == 99.).all()))

    def test_atomic_subset_preserves_selected_gate_rows_and_optimizer_state(self):
        model = self.trained()
        with self.assertRaisesRegex(ValueError, 'atomically'):
            model.patch_subset(torch.tensor([0, 1]))
        selected = model.patch_subset(torch.tensor([4, 3]))
        self.assertEqual(len(selected.patch_rows), 1)
        self.assertEqual(selected.owned_old_rows.tolist(), [1, 4])
        torch.testing.assert_close(selected.patch_retention, model.patch_retention[1:], rtol=0, atol=0)
        torch.testing.assert_close(selected.patch_uv, model.patch_uv[[4, 3]], rtol=0, atol=0)
        groups = {g['name']: g for g in selected.optimizer.param_groups}
        original = {g['name']: g for g in model.optimizer.param_groups}
        gate_state = selected.optimizer.state[groups['patch_retention']['params'][0]]['exp_avg']
        old_gate_state = model.optimizer.state[original['patch_retention']['params'][0]]['exp_avg']
        torch.testing.assert_close(gate_state, old_gate_state[1:], rtol=0, atol=0)
        selected.patch_uv.data.add_(.001)
        self.assertFalse(torch.equal(selected.patch_uv, model.patch_uv[[4, 3]]))

    def test_export_reinstall_retains_local_sheet_and_material_prior_for_refinement(self):
        model = self.trained()
        exported = model.export_patch_specs()
        rebuilt = SurfacePatchModel(1).install_patches(exported, old_count=6)
        for name in RAW_FIELDS:
            torch.testing.assert_close(model_raw(rebuilt)[name], model_raw(model)[name], rtol=0, atol=2.e-16)
        torch.testing.assert_close(rebuilt.patch_prior_dc, model.patch_prior_dc, rtol=0, atol=0)
        torch.testing.assert_close(rebuilt.patch_retention, model.patch_retention, rtol=0, atol=0)

    def test_empty_selection_has_no_hidden_old_coverage_effect(self):
        model = self.trained()
        empty = model.patch_subset(torch.zeros(5, dtype=torch.bool))
        self.assertEqual(len(empty._xyz), 0)
        self.assertEqual(len(empty.patch_rows), 0)
        torch.testing.assert_close(empty.raw_old_alpha(reference()['opacity']), reference()['opacity'].sigmoid(), rtol=0, atol=0)


# ----- test_tracked_surface_proposal -----
class TrackedSurfaceTests(unittest.TestCase):
 def test_correct_half_pixel_center_including_odd_width(self):
  got=colmap_half_pixel_coordinates([[1344.5,1008.],[.5,.5]],(2689,2016),(1345,1008))
  np.testing.assert_allclose(got[0],[672.,503.5],atol=1e-12)
  np.testing.assert_allclose(got[1],[.5*1345/2689-.5,-.25],atol=1e-12)

 def test_skip_original_xyz_cam00_and_ambiguous_observations(self):
  with tempfile.TemporaryDirectory() as directory:
   p=Path(directory)/'points3D.bin'
   observations=[[(1,0),(2,0),(3,0),(4,0)],[(2,1),(2,2),(3,1),(4,1)],[(1,2),(2,3),(3,3)]]
   with open(p,'wb') as f:
    f.write(struct.pack('<Q',len(observations)))
    for i,obs in enumerate(observations):
     f.write(struct.pack('<QdddBBBdQ',i+10,float('nan'),float('inf'),-1e30,255,0,0,1e99,len(obs)))
     for cam,index in obs:f.write(struct.pack('<ii',cam,index))
   tracks,report=_tracks(p,{2,3,4})
   self.assertEqual(len(tracks),1);self.assertEqual(tracks[0][0],10)
   self.assertEqual(tracks[0][1][:,0].tolist(),[2,3,4]);self.assertEqual(report['ambiguous_train_tracks_skipped'],1)
   self.assertFalse(report['original_xyz_used'])

 def test_current_train_DLT_and_covariance_without_initial_xyz(self):
  xyz=np.array([.2,-.1,3.]);k=np.array([[700.,0,672.],[0,700.,503.5],[0,0,1.]])
  ps={i:np.column_stack((np.eye(3),[-x,0,0])) for i,x in enumerate((-.4,0,.4))};ks={i:k for i in ps}
  obs={i:_project(xyz[None],p,k)[0][0] for i,p in ps.items()}
  got,cov,error,angle=_triangulate(obs,ps,ks)
  np.testing.assert_allclose(got,xyz,atol=1e-12);self.assertLess(error,1e-10)
  self.assertGreater(angle,2);self.assertTrue((np.linalg.eigvalsh(cov)>0).all())

 def test_active_binding_does_not_eat_fresh_quota_or_rebirth(self):
  p=[dict(uid='old',score=0,binding_only=True),dict(uid='fresh-low',score=1,binding_only=False),dict(uid='fresh-high',score=2,binding_only=False)]
  result,fresh,active=_select(p,{'old'},1)
  self.assertEqual([x['uid'] for x in result],['fresh-high','old']);self.assertEqual((fresh,active),(1,1))

 def test_actual_supported_residual_not_bbox_area_defines_score(self):
  r=np.zeros((8,8,3),np.float32);r[2,3]=[.1,.2,.3];r[7,7]=100
  score=_residual_score(r,np.array([[3.,2.],[3.,3.]],np.float32))
  self.assertAlmostEqual(score,.14,places=6)
  # A globally consistent peer view may have no pixel in the chosen connected
  # photo-support component; it contributes zero rather than a remap failure.
  self.assertEqual(_residual_score(r,np.empty((0,2),np.float32)),0.)

 def test_full_empty_supply_never_reads_cam00_or_db(self):
  camera=NS(model='PINHOLE',width=80,height=64,params=np.array([50.,50.,40.,32.]))
  images={i: NS(name=f'cam{i:02d}.png',camera_id=1,qvec=np.array([1.,0.,0.,0.]),tvec=np.zeros(3),xys=np.zeros((0,2))) for i in range(21)}
  read=[]
  def image(path,flags):
   self.assertNotIn('cam00',path);read.append(path);return np.zeros((32,40,3),np.uint8)
  with patch('libgs.data.utils.colmap.read_extrinsics_binary',return_value=images),patch('libgs.data.utils.colmap.read_intrinsics_binary',return_value={1:camera}),patch('pipeline.model.tracked_surface_proposal._tracks',return_value=([],dict(original_xyz_used=False))),patch('pipeline.model.tracked_surface_proposal.cv2.imread',side_effect=image):
   specs,report=propose_tracked_surface_births(2,None,'sparse',0,image_root='frames')
  self.assertEqual(specs,[]);self.assertEqual(len(read),40)
  self.assertFalse(report['external_test_images_read']);self.assertFalse(report['DB_used'])
  self.assertEqual(report['SGD_steps'],0);self.assertEqual(report['BA_steps'],0)
  with self.assertRaises(ValueError):propose_tracked_surface_births(2,{'cam00.png':None},'sparse',1,image_root='frames')


if __name__ == '__main__':
    unittest.main()
