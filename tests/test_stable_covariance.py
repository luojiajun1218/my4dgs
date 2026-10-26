import copy
import math
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from pipeline.model import contribution_rasterizer as raster
from pipeline.model import contribution_renderer as render
from pipeline import module as module_source
import test_codec as checkpoint_fixtures
import test_spatial_motion as gpu_fixtures


class StableCovarianceIntegrationTests(unittest.TestCase):
    def test_gradient_probes_use_stable_backend_and_inference_keeps_native_path(self):
        cfg=render.RendererConfig(stable_covariance_gradients=True)
        renderer=render.ContributionRenderer(cfg,torch.nn.Identity())
        native=SimpleNamespace(raster_settings=object())
        with patch.object(render.Renderer,'create_rasterizer',return_value=native), \
                patch.object(render,'_CollectingRasterizer',return_value='stable') as wrapper:
            self.assertEqual(renderer.create_rasterizer(None,None,1.),'stable')
            self.assertTrue(wrapper.call_args.args[-1])
            with torch.no_grad():
                self.assertIs(renderer.create_rasterizer(None,None,1.),native)
                renderer.collect_contributions=True
                self.assertEqual(renderer.create_rasterizer(None,None,1.),'stable')
        renderer.config.stable_covariance_gradients=1
        with patch.object(render.Renderer,'create_rasterizer',return_value=native), \
                self.assertRaisesRegex(ValueError,'boolean'):
            renderer.create_rasterizer(None,None,1.)

    def test_build_cache_isolates_source_versions_paths_and_gradient_modes(self):
        raster._backend.cache_clear()
        self.addCleanup(raster._backend.cache_clear)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);extension=root/'extensions/observed_rasterizer'
            (extension/'cuda_rasterizer').mkdir(parents=True)
            header=extension/'cuda_rasterizer/backward.h';header.write_text('version one')
            fake_file=root/'pipeline/model/contribution_rasterizer.py'
            with patch.object(raster,'__file__',str(fake_file)), \
                    patch('torch.utils.cpp_extension.load',return_value=object()) as load:
                raster._backend(False);legacy=load.call_args.kwargs
                raster._backend(True);stable=load.call_args.kwargs
                self.assertNotEqual(legacy['name'],stable['name'])
                self.assertNotIn('-DMY4DGS_STABLE_COVARIANCE_GRADIENTS',legacy['extra_cuda_cflags'])
                self.assertIn('-DMY4DGS_STABLE_COVARIANCE_GRADIENTS',stable['extra_cuda_cflags'])
                header.write_text('version two');raster._backend.cache_clear()
                raster._backend(True)
                self.assertNotEqual(stable['name'],load.call_args.kwargs['name'])

    def test_legacy_checkpoint_missing_the_flag_resumes_only_with_legacy_configuration(self):
        cfg=module_source.ModuleConfig()
        gaussians=checkpoint_fixtures.FakeNativeGaussians()
        saved=asdict(cfg);saved['renderer'].pop('stable_covariance_gradients')
        packet=dict(gaussians=gaussians.capture(),rigid_tree_version=1,
            rigid_tree_config=asdict(cfg.rigid_tree),rigid_tree_module_config=saved,
            rigid_tree_frame=dict(prepared=False,finalized=True,motion_committed=False,
                reference_count=4,report=None,birth_control={}),births=None)
        holder=checkpoint_fixtures.fake_module(cfg,gaussians,frame=1)
        module_source.Module.on_load_checkpoint(holder,copy.deepcopy(packet))
        self.assertEqual(holder.reference_count,4)
        changed=copy.deepcopy(cfg);changed.renderer.stable_covariance_gradients=True
        holder=checkpoint_fixtures.fake_module(changed,gaussians,frame=1)
        with self.assertRaisesRegex(ValueError,'same module configuration'):
            module_source.Module.on_load_checkpoint(holder,packet)


@unittest.skipUnless(torch.cuda.is_available(),'Native CUDA derivative checks')
class StableCovarianceGPUChecks(unittest.TestCase):
    def test_precomputed_covariance_returns_float32_gradients_matching_finite_differences(self):
        settings=gpu_fixtures.MotionContributionGPUChecks().settings()
        xyz=torch.tensor([[0.,0.,2.]],device='cuda')
        covariance=torch.tensor([[.02,.003,.001,.01,.0005,.005]],device='cuda',requires_grad=True)
        engine=raster.GaussianRasterizer(settings,stable_covariance_gradients=True)
        def image():
            return engine(means3D=xyz,means2D=torch.zeros_like(xyz),
                opacities=torch.tensor([[.5]],device='cuda'),
                colors_precomp=torch.tensor([[.7,.4,.2]],device='cuda'),
                cov3D_precomp=covariance)[0]
        weight=torch.zeros(3,64,64,device='cuda',dtype=torch.double)
        weight[:,30:34,30:34]=1/12288
        gradient,=torch.autograd.grad((image().double()*weight).sum(),covariance)
        self.assertEqual(gradient.dtype,torch.float32)
        self.assertEqual(gradient.shape,covariance.shape)
        with torch.no_grad():
            original=covariance.clone();direction=gradient/gradient.norm();step=1e-4
            covariance.copy_(original+step*direction);plus=image()
            covariance.copy_(original-step*direction);minus=image()
            numerical=float(((plus.double()-minus.double())*weight).sum()/(2*step))
            analytical=float((gradient*direction).sum())
        self.assertLess(abs(numerical-analytical)/max(abs(numerical),abs(analytical),1e-9),.03)

    def test_actual_render_derivatives_match_finite_differences_for_all_fitted_fields(self):
        settings=gpu_fixtures.MotionContributionGPUChecks().settings()
        parameters={
            'xyz':torch.tensor([[.07,-.03,2.]],device='cuda',requires_grad=True),
            'logscale':torch.tensor([[.12,.08,.04]],device='cuda').log().requires_grad_(),
            'rotation':torch.tensor([[1.,.1,.08,-.04]],device='cuda',requires_grad=True),
            'logit':torch.tensor([[.2]],device='cuda',requires_grad=True),
            'sh':torch.tensor([[[.5,.2,.1],[.02,-.03,.01],[-.04,.01,.03],[.03,.02,-.01]]],
                              device='cuda',requires_grad=True)}
        stable=raster.GaussianRasterizer(settings,stable_covariance_gradients=True)
        def image():
            return stable(means3D=parameters['xyz'],means2D=torch.zeros_like(parameters['xyz']),
                shs=parameters['sh'],opacities=parameters['logit'].sigmoid(),
                scales=parameters['logscale'].exp(),
                rotations=torch.nn.functional.normalize(parameters['rotation'],dim=-1))[0]
        yy,xx=torch.meshgrid(torch.arange(64,device='cuda'),torch.arange(64,device='cuda'),indexing='ij')
        homogeneous=torch.cat((parameters['xyz'].detach(),torch.ones(1,1,device='cuda')),1)@settings.projmatrix
        center=(homogeneous[0,:2]/homogeneous[0,3]+1)*32-.5
        mask=(xx-center[0]).square()+(yy-center[1]).square()<4
        weight=(torch.sin(xx*.13)+torch.cos(yy*.19)+.3)*mask
        weight=torch.stack((weight,.4*weight,-.2*weight)).double()/12288
        loss=(image().double()*weight).sum()
        gradients=torch.autograd.grad(loss,list(parameters.values()))
        for (name,parameter),gradient in zip(parameters.items(),gradients):
            with self.subTest(field=name),torch.no_grad():
                direction=gradient/gradient.norm().clamp_min(1e-30)
                original=parameter.clone();step=.0005 if name=='xyz' else .002
                parameter.copy_(original+step*direction);plus=image()
                parameter.copy_(original-step*direction);minus=image()
                parameter.copy_(original)
                numerical=float(((plus.double()-minus.double())*weight).sum()/(2*step))
                analytic=float((gradient*direction).sum())
                self.assertLess(abs(numerical-analytic)/max(abs(numerical),abs(analytic),1e-9),.03)

    def test_ill_conditioned_splat_keeps_exact_image_and_stable_repeated_position_gradient(self):
        from diff_gaussian_rasterization import GaussianRasterizer
        settings=gpu_fixtures.MotionContributionGPUChecks().settings()
        xyz=torch.tensor([[0.,0.,1.]],device='cuda',requires_grad=True)
        angle=.37;half=math.sqrt(.5)
        q=torch.tensor([[math.cos(angle/2)*half,math.cos(angle/2)*half,
                         math.sin(angle/2)*half,math.sin(angle/2)*half]],device='cuda')
        args=dict(means3D=xyz,means2D=torch.zeros_like(xyz),shs=None,
            colors_precomp=torch.tensor([[.7,.4,.2]],device='cuda'),
            opacities=torch.tensor([[.2]],device='cuda'),
            scales=torch.tensor([[20.,1.,1e-7]],device='cuda'),rotations=q)
        native=GaussianRasterizer(settings)(**args)[0]
        actual=raster.GaussianRasterizer(settings,stable_covariance_gradients=True)(**args)[0]
        self.assertTrue(torch.equal(actual,native))
        yy,xx=torch.meshgrid(torch.arange(64,device='cuda'),torch.arange(64,device='cuda'),indexing='ij')
        weight=(torch.sin(.13*xx+.19*yy)+.4).expand_as(actual)/12288
        gradients=[torch.autograd.grad(actual,xyz,grad_outputs=weight,retain_graph=True)[0]
                   for _ in range(8)]
        values=torch.stack(gradients);mean=values.mean(0)
        self.assertTrue(bool(torch.isfinite(values).all()))
        self.assertGreater(float(mean.norm()),1e-8)
        self.assertLess(float((values-mean).norm(dim=-1).max()/mean.norm()),.002)


if __name__=='__main__':
    unittest.main()
