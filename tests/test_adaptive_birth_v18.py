"""CPU diagnostics for storage pressure and conservative two-view colours."""
import math
import tempfile
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch
import torch
from tests.test_reincarnation import early_model_fixture, M
from tests.test_adaptive_birth import controller_fixture
from pipeline.model.adaptive_birth_model import AdaptiveBirthModel
from pipeline.model.birth_quantization import QuantizedNewGaussianView
from pipeline.model.adaptive_birth_storage import newborn_storage_terms, storage_ramp
from pipeline.model.adaptive_birth_colour import conservative_colour_seed
from pipeline.model.support_reincarnation_proposal import _native_responsibility


def storage_fixture(alpha=.1):
    old,raw=early_model_fixture()
    model=AdaptiveBirthModel(1).install(old._seed_raw,old.parent_rows)
    with torch.no_grad():
        model._opacity.fill_(alpha)
        model.local_xyz.fill_(.003);model.local_rotvec.fill_(.02)
        model._scaling.add_(.03);model._features_dc.add_(.05);model._features_rest.add_(.02)
    cfg=NS(new_xyz_step_extent=1e-5,new_rotation_step=.001,new_logscale_step=.001,
           new_opacity_step=.01,new_sh_step=.001,rotation_updates=True)
    return model,QuantizedNewGaussianView(model,cfg,2.,raw,model.parent_rows)


class NewbornStorageTests(unittest.TestCase):
    def test_alpha_exit_gradient_on_both_sides_of_parent_and_threshold(self):
        for alpha in (0.,.001,.005,.009,.01,.03,.1,.5,.9,1.):
            with self.subTest(alpha=alpha):
                model,view=storage_fixture(alpha)
                terms=newborn_storage_terms(view)
                terms['surrogate_bytes'].backward()
                self.assertTrue(torch.isfinite(model._opacity.grad).all())
                self.assertTrue(bool((model._opacity.grad>0).all()))
                if alpha:
                    for attr in ('local_xyz','local_rotvec','_scaling','_features_dc','_features_rest'):
                        gradient=getattr(model,attr).grad
                        self.assertTrue(torch.isfinite(gradient).all(),attr)
                        self.assertGreater(float(gradient.abs().sum()),0,attr)
                    self.assertTrue(bool((model._features_dc.grad>0).all()))
                    self.assertTrue(bool((model._features_rest.grad>0).all()))
                    self.assertGreater(float((model.local_xyz.grad*model.local_xyz).sum()),0)

    def test_ramp_exact_boundaries_and_same_price_normalization(self):
        self.assertEqual([storage_ramp(s) for s in (70,90,91,125,160,200,300)],
                         [0.,0.,1/70,.5,1.,1.,1.])
        with tempfile.TemporaryDirectory() as directory:
            m,_=controller_fixture(directory)
            m.config.birth.storage_loss_enabled=True
            m.config.rigid_tree.rate_loss_weight=.002
            penalty,metrics=m._adaptive_storage_regularization(160)
            expected=.002/m.motion.rate_normalization_bytes
            torch.testing.assert_close(penalty,metrics['birth_storage_surrogate_bytes']*expected)
            self.assertEqual(m._adaptive_storage_regularization(90)[0].item(),0.)

    def test_source_unsigned_varint_boundaries_and_empty_birth(self):
        model,view=storage_fixture()
        for source,width in ((0,1),(127,1),(128,2),(16383,2),(16384,3),(2097151,3),(2097152,4)):
            view.source=torch.full_like(view.source,source)
            terms=newborn_storage_terms(view)
            # SH1: 3 scale + 1 alpha + 3 DC + 9 SH = 16 int16s.
            expected=terms['support_points']*(width+7+32)
            torch.testing.assert_close(terms['base_bytes'],expected)
        with tempfile.TemporaryDirectory() as directory:
            m,_=controller_fixture(directory,empty=True)
            self.assertEqual(m._adaptive_storage_regularization(160)[0].item(),0.)

    def test_native_colour_evidence_reuses_four_renders_and_restores_renderer(self):
        with tempfile.TemporaryDirectory() as directory:
            m,views=controller_fixture(directory)
            raw,alpha=m._early_current_old(70)
            from pipeline.model.residual_birth import RawView
            old=RawView(raw,m.config.sh_degree,m.gaussians,alpha)
            proposal=NS(gaussians=old,renderer=m.renderer,background=m.background)
            original=m.renderer.gaussians
            renderer=type(m.renderer).__call__
            with patch.object(type(m.renderer),'__call__',autospec=True,side_effect=renderer) as renders:
                result=_native_responsibility(proposal,dict(list(views.items())[:2]),collect_colour_evidence=True)
                self.assertEqual(renders.call_count,4)
            self.assertIs(m.renderer.gaussians,original)
            self.assertEqual(result[3]['native_colour_backward_calls'],6)
            for item in result[4]:
                self.assertTrue(bool((item['native_mass']>=0).all()))
                self.assertEqual(item['signed_response'].shape,(len(raw['xyz']),3))

    def test_rate_has_no_parent_motion_gradient_and_forward_price_uses_render_symbols(self):
        model,view=storage_fixture()
        parent_xyz=model._source_xyz.detach().clone().requires_grad_(True)
        parent_q=model._source_q.detach().clone().requires_grad_(True)
        parent_xyz.data.add_(.7)
        model._source_xyz=parent_xyz;model._source_q=parent_q
        # The actual renderer supplies current moved predictors, not frame-start
        # coordinates. Its forward q remains the small child-parent residual.
        reference={name:value.clone() for name,value in view.predictor.items()}
        reference['xyz']=parent_xyz.detach();reference['rotation']=parent_q.detach()
        current=QuantizedNewGaussianView(model,view.config,2.,reference,torch.arange(len(parent_xyz)))
        terms=newborn_storage_terms(current)
        self.assertLess(float(current.symbols['xyz'].abs().max()),500)
        terms['surrogate_bytes'].backward()
        self.assertIsNone(parent_xyz.grad)
        self.assertIsNone(parent_q.grad)
        self.assertGreater(float(model.local_xyz.grad.abs().sum()),0)
        self.assertGreater(float(model.local_rotvec.grad.abs().sum()),0)

    def test_real_training_uses_render_view_and_two_views_charge_once(self):
        from types import MethodType
        from tests.test_reincarnation import T
        with tempfile.TemporaryDirectory() as directory:
            m,views=controller_fixture(directory)
            m.config.birth.storage_loss_enabled=True;m.config.rigid_tree.rate_loss_weight=.002
            m.trainer.global_step=160
            camera=next(iter(views.values()))
            _,metrics,package=M.Module.training_step(m,camera,160,regularization_weight=1.)
            self.assertNotIn('_adaptive_birth_rate_view',package)
            expected=m._adaptive_storage_regularization(160,
                m.forward(camera,training=True)['_adaptive_birth_rate_view'])[0]
            self.assertGreater(float(metrics['birth_storage_loss']),0)
            torch.testing.assert_close(metrics['birth_storage_loss'],expected)
            m.training_step=MethodType(M.Module.training_step,m)
            trainer=NS(module=m,config=NS(motion_views_per_step=2))
            trainer._views_per_step=MethodType(T.Trainer._views_per_step,trainer)
            m.early_shared_steps=159;m.early_birth_updates=89
            with patch.object(m,'_adaptive_storage_regularization',wraps=m._adaptive_storage_regularization) as rate:
                _,averaged=T.Trainer._fit_views(trainer,NS(next=lambda:camera),160,defer_metrics=True)
                self.assertEqual(rate.call_count,1)
            torch.testing.assert_close(averaged['birth_storage_loss'],expected,rtol=1e-5,atol=1e-10)


def colour_fixture(signs=(1.,1.),mass=5.):
    xyz=torch.tensor([[0.,0.,3.]])
    raw=dict(xyz=xyz,rotation=torch.tensor([[1.,0.,0.,0.]]),logscale=torch.full((1,3),math.log(.2)),
             opacity=torch.zeros(1,1),sh_dc=torch.zeros(1,1,3),sh_rest=torch.zeros(1,3,3))
    seed={k:v.clone() for k,v in raw.items()}
    cache=[]
    for i,sign in enumerate(signs):
        residual=torch.full((3,32,32),-.01*sign)
        view=NS(world_view_transform=torch.eye(4),fovx=math.pi/2,fovy=math.pi/2,path=f'cam{i+1:02d}.png')
        cache.append(dict(view=view,name=view.path,residual=residual,observed=torch.ones(1,dtype=torch.bool),
                          native_mass=torch.tensor([mass]),signed_response=torch.full((1,3),mass*.01*sign)))
    return raw,seed,torch.tensor([0]),torch.tensor([[.1]]),cache


class NewbornColourTests(unittest.TestCase):
    def test_coherent_small_offset_and_conflict_weak_support_fallback(self):
        for signs,mass,accept in (((1.,1.),5.,True),((1.,-1.),5.,False),((1.,1.),1e-6,False)):
            args=colour_fixture(signs,mass)
            report=conservative_colour_seed(*args)
            self.assertEqual(report['accepted'],int(accept))
            offset=(args[1]['sh_dc']-args[0]['sh_dc'])*.28209479177387814
            self.assertLessEqual(float(offset.abs().max()),.02500001)
            self.assertEqual(bool(offset.abs().sum()>0),accept)
            for field in ('xyz','rotation','logscale','opacity','sh_rest'):
                torch.testing.assert_close(args[0][field],args[1][field],rtol=0,atol=0)

    def test_test_camera_rejected_and_large_residual_clamped(self):
        args=colour_fixture()
        args[-1][0]['name']='cam00.png'
        with self.assertRaisesRegex(ValueError,'held-out'):conservative_colour_seed(*args)
        args=colour_fixture()
        for item in args[-1]:
            item['residual'].fill_(-100);item['signed_response'].fill_(500)
        report=conservative_colour_seed(*args)
        self.assertEqual(report['accepted'],1)
        self.assertLessEqual(report['max_applied_rgb_delta'],.02500001)

    def test_configuration_defaults_keep_v17_and_validate(self):
        cfg=M.BirthConfig()
        self.assertFalse(cfg.storage_loss_enabled)
        self.assertFalse(cfg.residual_colour_init)
        cfg.storage_loss_enabled=True
        with self.assertRaisesRegex(ValueError,'adaptive_residual_birth'):cfg.validate(200)


if __name__=='__main__':unittest.main()
