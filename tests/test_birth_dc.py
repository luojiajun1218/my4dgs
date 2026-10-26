"""Physical box, coupled solve and real colour-Jacobian call contract."""
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

from pipeline.model.adaptive_birth_dc import native_colour_seed,projected_colour_fit
from pipeline.model.reincarnation_model import C0


class BoxSolveTests(unittest.TestCase):
    def test_known_coupled_box_optimum_and_disabled_channel(self):
        h=torch.tensor([[4.,1.],[1.,2.]])
        rhs=torch.tensor([[1.,1.,1.],[-1.,-1.,-1.]])
        lower=torch.tensor([[0.,-1.,-1.],[-.5,-1.,-1.]])
        upper=torch.tensor([[.3,1.,1.],[0.,1.,1.]])
        active=torch.ones_like(rhs,dtype=torch.bool);active[0,2]=False
        result,report=projected_colour_fit(rhs,lambda x:h@x,lower,upper,active,64)
        expected=torch.tensor([[.3,3/7,0.],[-.5,-5/7,-.5]])
        torch.testing.assert_close(result,expected,atol=2e-5,rtol=2e-5)
        self.assertEqual(report['active_channels'],5)
        objective=lambda x:.5*(x*(h@x)).sum()-(rhs*x).sum()
        self.assertLess(float(objective(result)),float(objective(torch.zeros_like(result))))

    def test_no_active_channels_do_not_render_and_invalid_zero_box_rejected(self):
        x=torch.ones(2,3)
        def forbidden(value):raise AssertionError('No native work is needed')
        result,report=projected_colour_fit(x,forbidden,-x,x,x.bool()&False)
        self.assertTrue(torch.equal(result,torch.zeros_like(x)))
        self.assertEqual(report['normal_operator_calls'],0)
        with self.assertRaisesRegex(ValueError,'zero offset'):
            projected_colour_fit(x,forbidden,x,x*2,x.bool())


class LinearRenderer:
    def __init__(self):
        self.gaussians='original';self.collect_contributions=True
        self.last_contributions='original';self.calls=0

    def __call__(self,view,background,override_color):
        self.calls+=1
        return {'render':(view.weights@override_color).T[:,None,:]}


def fixture(opposed=False,invalid_sphere=False):
    def raw(rgb):
        count=len(rgb)
        dc=((torch.tensor(rgb,dtype=torch.float32)-.5)/C0/.001).round()*.001
        return dict(xyz=torch.tensor([[.1*i,0.,2.] for i in range(count)]),
                    rotation=torch.tensor([[1.,0.,0.,0.]]).repeat(count,1),
                    logscale=torch.full((count,3),-2.),opacity=torch.zeros(count,1),
                    sh_dc=dc[:,None,:],sh_rest=torch.zeros(count,3,3))
    old=raw([[.4,.4,.4]]);new=raw([[.3,.3,.3],[.4,.4,.4]])
    if invalid_sphere:new['sh_rest'].fill_(3.)
    current=torch.cat((old['sh_dc'],new['sh_dc']))[:,0]*C0+.5
    target=current.clone();target[1]=.65;target[2]=.2
    views=[]
    for i in range(2):
        weights=torch.tensor([[.5,.3,.1],[.4,.2,.4],[.6,.1,.05]])
        if i:weights=weights[[1,2,0]]
        desired=target if not(opposed and i) else current*2-target
        image=(weights@desired).T[:,None,:]
        views.append(SimpleNamespace(path=Path(f'cam0{i+1}.png'),camera_center=torch.zeros(3),image=image,weights=weights))
    return old,new,views


class NativeDCSeedTests(unittest.TestCase):
    def test_joint_overlap_fit_reduces_error_and_keeps_parent_quantization(self):
        old,new,views=fixture();renderer=LinearRenderer()
        template=SimpleNamespace(active_sh_degree=1,max_sh_degree=1)
        before={k:v.clone() for k,v in new.items()}
        dc,report=native_colour_seed(renderer,template,old,new,views,torch.zeros(3),
            dc_step=.001,predictor_dc=torch.zeros_like(new['sh_dc']))
        colors=dc[:,0]*C0+.5
        # This is a fixed16-iteration initializer, not a converged optimizer.
        torch.testing.assert_close(colors,torch.tensor([[.65]*3,[.2]*3]),atol=.005,rtol=0)
        initial=torch.cat((old['sh_dc'],new['sh_dc']))[:,0]*C0+.5
        final=torch.cat((initial[:1],colors))
        before_error=sum((view.weights@initial-view.image[:,0].T).square().sum() for view in views)
        after_error=sum((view.weights@final-view.image[:,0].T).square().sum() for view in views)
        self.assertLess(float(after_error),float(before_error)*.001)
        torch.testing.assert_close(dc/.001,(dc/.001).round(),atol=1e-4,rtol=0)
        self.assertEqual(renderer.calls,report['additional_render_calls'])
        self.assertEqual(report['additional_render_calls'],36)
        self.assertEqual(report['additional_backward_calls'],36)
        self.assertEqual(renderer.gaussians,'original')
        self.assertEqual(renderer.last_contributions,'original')
        self.assertTrue(renderer.collect_contributions)
        for name in new:torch.testing.assert_close(new[name],before[name],atol=0,rtol=0)

    def test_opposed_views_and_invalid_global_sh_channels_cannot_change(self):
        template=SimpleNamespace(active_sh_degree=1,max_sh_degree=1)
        for options in (dict(opposed=True),dict(invalid_sphere=True)):
            old,new,views=fixture(**options)
            dc,report=native_colour_seed(LinearRenderer(),template,old,new,views,torch.zeros(3),
                dc_step=.001,predictor_dc=torch.zeros_like(new['sh_dc']))
            torch.testing.assert_close(dc,new['sh_dc'],atol=0,rtol=0)
            self.assertEqual(report['accepted'],0)
            self.assertEqual(report['additional_render_calls'],4)

    def test_test_camera_cannot_enter_colour_fit(self):
        old,new,views=fixture();views[0].path=Path('cam00.png')
        with self.assertRaisesRegex(ValueError,'test cameras'):
            native_colour_seed(LinearRenderer(),SimpleNamespace(),old,new,views,torch.zeros(3),
                               dc_step=.001,predictor_dc=torch.zeros_like(new['sh_dc']))


class BirthDCControllerTests(unittest.TestCase):
    def test_two_view_initializer_is_recorded_before_shared_adam_attachment(self):
        import tempfile
        from unittest.mock import patch
        from pipeline.early_replacement import EarlyReplacementController
        from tests.test_adaptive_birth import controller_fixture
        original=EarlyReplacementController._early_current_old
        def enable(module,step):
            module.config.birth.native_dc_init=True
            return original(module,step)
        selected=[]
        def initialize(renderer,template,old,newborn,views,background,*,dc_step,predictor_dc):
            self.assertEqual(len(views),2)
            self.assertNotIn('cam00.png',views)
            symbols=((newborn['sh_dc']-predictor_dc)/dc_step).round()+2
            dc=predictor_dc+symbols*dc_step;selected.append(dc.clone())
            return dc,dict(enabled=True,accepted=len(dc),additional_render_calls=36,additional_backward_calls=36)
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(EarlyReplacementController,'_early_current_old',enable), \
                patch('pipeline.adaptive_birth.native_colour_seed',side_effect=initialize) as solver:
            module,_=controller_fixture(directory)
            self.assertEqual(solver.call_count,1)
            self.assertIs(module.births.optimizer,module.motion.optimizer)
            for value in (module.births._features_dc,module.births._seed_raw['sh_dc'],module.birth_transaction.raw['sh_dc']):
                torch.testing.assert_close(value,selected[0],atol=0,rtol=0)
            report=module.birth_control['early_replacement']
            self.assertEqual(report['proposal_render_calls'],40)
            self.assertEqual(report['native_colour_initialization']['additional_backward_calls'],36)
            restored=type(module.births)(1).restore(module.births.capture())
            torch.testing.assert_close(restored._features_dc,selected[0],atol=0,rtol=0)

    def test_legacy_default_and_explicit_adaptive_mode_validation(self):
        from pipeline.module import BirthConfig
        config=BirthConfig();self.assertFalse(config.native_dc_init)
        config.native_dc_init=True
        with self.assertRaisesRegex(ValueError,'adaptive_residual_birth'):config.validate(200)
        config.radiance_mode='adaptive_residual_birth';config.validate(200)
        config.native_dc_init=1
        with self.assertRaisesRegex(ValueError,'boolean'):config.validate(200)


@unittest.skipUnless(torch.cuda.is_available(),'Native two-view colour initialization')
class BirthDCNativeGPUChecks(unittest.TestCase):
    def test_actual_native_and_stable_training_backends_keep_the_same_colour_solution(self):
        import math
        from libgs.renderer import Renderer,RendererConfig
        from pipeline.model.contribution_renderer import ContributionRenderer,RendererConfig as TrainingConfig
        from pipeline.model.gaussian import GaussianModel
        from pipeline.model.residual_birth import RawView
        from tests.test_spatial_motion import MotionContributionGPUChecks
        old,new,_=fixture()
        old={k:v.cuda() for k,v in old.items()};new={k:v.cuda() for k,v in new.items()}
        basis=GaussianModel(1);basis.active_sh_degree=1
        settings=MotionContributionGPUChecks().settings()
        target={**new,'sh_dc':((torch.tensor([[.65]*3,[.2]*3],device='cuda')-.5)/C0)[:,None]}
        full={k:torch.cat((old[k],target[k])) for k in old}
        native=Renderer(RendererConfig(),basis);native.gaussians=RawView(full,1,basis)
        background=torch.zeros(3,device='cuda');views=[]
        for index in range(2):
            view=SimpleNamespace(path=Path(f'cam0{index+1}.png'),image=torch.zeros(3,64,64,device='cuda'),
                fovx=2*math.atan(settings.tanfovx),fovy=2*math.atan(settings.tanfovy),
                world_view_transform=settings.viewmatrix,full_proj_transform=settings.projmatrix,camera_center=settings.campos)
            with torch.no_grad():view.image=native(view,background)['render'].detach()
            views.append(view)
        answers=[]
        for renderer in (native,ContributionRenderer(TrainingConfig(stable_covariance_gradients=True),basis)):
            dc,report=native_colour_seed(renderer,basis,old,new,views,background,
                dc_step=.001,predictor_dc=torch.zeros_like(new['sh_dc']))
            before={k:torch.cat((old[k],new[k])) for k in old}
            after={**before,'sh_dc':torch.cat((old['sh_dc'],dc))}
            errors=[]
            with torch.no_grad():
                for raw in (before,after):
                    native.gaussians=RawView(raw,1,basis)
                    errors.append(sum((native(view,background)['render']-view.image).square().sum() for view in views))
            self.assertLess(float(errors[1]),float(errors[0])*.1)
            self.assertGreater(report['accepted'],0)
            answers.append(dc)
        torch.testing.assert_close(answers[0],answers[1],atol=.0011,rtol=0)


if __name__=='__main__':unittest.main()
