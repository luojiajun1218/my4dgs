"""Exact image moments and coordinate/covariance checks without a renderer."""
import math
from types import SimpleNamespace
import unittest
import torch
from pipeline.model.support_reincarnation_proposal import _native_responsibility,_rotation
from pipeline.model.residual_support_geometry import residual_support_seed,residual_support_opacity,support_distribution,_matrix_quaternion
from pipeline.module import BirthConfig


class LinearRenderer:
    def __init__(self,weights,colours,model):
        self.weights=weights;self.colours=colours;self.gaussians=model
        self.collect_contributions=False;self.calls=0
    def __call__(self,view,bg,override_color=None):
        self.calls+=1
        colour=self.colours if override_color is None else override_color
        return dict(render=torch.einsum('nhw,nc->chw',self.weights,colour),
            motion_contribution_filter=self.weights.sum((1,2)).gt(0))


class NativeMomentTests(unittest.TestCase):
    def fixture(self):
        weights=torch.tensor([[[.4,.2,0],[.1,0,0]],[[0,0,0],[0,0,.6]]])
        colours=torch.tensor([[.2,.3,.4],[.7,.1,.2]])
        model=SimpleNamespace(get_xyz=torch.zeros(2,3))
        renderer=LinearRenderer(weights,colours,model)
        module=SimpleNamespace(gaussians=model,renderer=renderer,background=torch.zeros(3))
        view=SimpleNamespace(path='cam01.png',image=torch.zeros(3,2,3))
        return module,view,weights,colours
    def test_moments_match_actual_visibility_weights_without_extra_forward(self):
        module,view,w,colour=self.fixture()
        score,_,_,report,cache=_native_responsibility(module,[view],collect_colour_evidence=True,collect_support_moments=True)
        image=torch.einsum('nhw,nc->chw',w,colour);energy=image.square().sum(0)
        x=torch.tensor([[-2/3,0.,2/3],[-2/3,0.,2/3]])
        y=torch.tensor([[-.5,-.5,-.5],[.5,.5,.5]])
        expected_first=torch.stack([(w*energy).sum((1,2)),(w*energy*x).sum((1,2)),(w*energy*y).sum((1,2))],1)
        expected_second=torch.stack([(w*energy*x*x).sum((1,2)),(w*energy*x*y).sum((1,2)),(w*energy*y*y).sum((1,2))],1)
        torch.testing.assert_close(cache[0]['support_first_moments'],expected_first)
        torch.testing.assert_close(cache[0]['support_second_moments'],expected_second)
        torch.testing.assert_close(score,expected_first[:,0])
        self.assertEqual(module.renderer.calls,2)
        self.assertEqual(report['native_colour_backward_calls'],4)
        self.assertEqual(report['support_extra_render_calls'],0)
        self.assertFalse(module.renderer.collect_contributions)
    def test_default_scores_and_colour_evidence_remain_compatible(self):
        a,v,_,_=self.fixture();b,_,_,_=self.fixture()
        old=_native_responsibility(a,[v],collect_colour_evidence=True)
        new=_native_responsibility(b,[v],collect_colour_evidence=True,collect_support_moments=True)
        torch.testing.assert_close(old[0],new[0])
        for key in ('native_mass','signed_response'):
            torch.testing.assert_close(old[4][0][key],new[4][0][key])
        self.assertNotIn('support_first_moments',old[4][0])
        self.assertEqual(old[3]['native_colour_backward_calls'],3)
    def test_black_residual_has_no_mass_or_nonfinite_moments(self):
        module,v,_,_=self.fixture();module.renderer.colours.zero_()
        score,_,_,_,cache=_native_responsibility(module,[v],collect_support_moments=True)
        self.assertFalse(score.any());self.assertFalse(cache[0]['support_first_moments'].any())
        self.assertTrue(torch.isfinite(cache[0]['support_second_moments']).all())
    def test_birth_jump_uses_raw_sse_separate_from_floor_weighted_scores(self):
        module,v,w,colour=self.fixture()
        _,_,_,report,_=_native_responsibility(module,[v],residual_floor=.03)
        raw=torch.einsum('nhw,nc->chw',w,colour).square()
        mse=raw.mean(0,keepdim=True)
        filtered=raw*((mse-.03**2).clamp_min(0)/mse.clamp_min(1e-30))
        self.assertAlmostEqual(report['raw_old_RGB_SSE'],float(raw.sum()),places=7)
        self.assertAlmostEqual(report['old_RGB_SSE'],float(filtered.sum()),places=7)
        self.assertGreater(report['raw_old_RGB_SSE'],report['old_RGB_SSE'])


def view(name='cam01.png',matrix=None,size=100):
    return SimpleNamespace(path=name,image=torch.zeros(3,size,size),fovx=math.pi/2,fovy=math.pi/2,
        world_view_transform=torch.eye(4) if matrix is None else matrix)


def evidence(camera,mean=(.2,-.1),covariance=((.0025,0),(0,.0004)),mass=1.):
    m=torch.tensor(mean,dtype=torch.float64);c=torch.tensor(covariance,dtype=torch.float64)
    first=torch.tensor([[mass,mass*m[0],mass*m[1]]],dtype=torch.float64)
    second=mass*torch.stack((c[0,0]+m[0]**2,c[0,1]+m[0]*m[1],c[1,1]+m[1]**2))[None]
    return dict(view=camera,name=camera.path,observed=torch.tensor([True]),
        support_first_moments=first,support_second_moments=second)


class SupportGeometryTests(unittest.TestCase):
    def raw(self):
        return dict(xyz=torch.tensor([[0.,0.,5.]]),rotation=torch.tensor([[1.,0.,0.,0.]]),
            logscale=torch.tensor([[20.,1e-6,20.]]).log())
    def test_measured_footprint_replaces_huge_parent_and_projects_correctly(self):
        raw=self.raw();cam=view();item=evidence(cam);q=1e-6
        seed,valid,report=residual_support_seed(raw,torch.tensor([0]),[item],q)
        self.assertTrue(valid.all());self.assertLess(float(seed['xyz'][0,2]),5.)
        self.assertLess(float(seed['logscale'].exp().max()),.3)
        self.assertEqual(report['single_view_count'],1)
        x,y,z=seed['xyz'][0].double()
        torch.testing.assert_close(torch.stack((x/z,y/z)),torch.tensor([.2,-.1],dtype=torch.float64),atol=1e-7,rtol=1e-6)
        r=_rotation(seed['rotation'].double());cov=(r*seed['logscale'].double().exp().square()[:,None])@r.transpose(-1,-2)
        jac=torch.tensor([[1/z,0,-x/z**2],[0,1/z,-y/z**2]],dtype=torch.float64)
        projected=jac@cov[0]@jac.T
        expected=torch.diag(torch.tensor([.0025+1/30000,.0004+1/30000],dtype=torch.float64))
        torch.testing.assert_close(projected,expected,atol=1e-9,rtol=3e-6)
    def test_camera_rotation_translation_and_scene_units_are_consistent(self):
        angle=.8;c,s=math.cos(angle),math.sin(angle)
        inverse=torch.eye(4,dtype=torch.float64);inverse[:3,:3]=torch.tensor([[c,0,-s],[0,1,0],[s,0,c]])
        inverse[3,:3]=torch.tensor([3.,-2.,1.])
        raw=self.raw();raw['xyz']=(raw['xyz'].double()@inverse[:3,:3]+inverse[3,:3]).float()
        cam=view(matrix=torch.linalg.inv(inverse));a,_,_=residual_support_seed(raw,torch.tensor([0]),[evidence(cam)],1e-6)
        camera=a['xyz'].double()@cam.world_view_transform[:3,:3]+cam.world_view_transform[3,:3]
        torch.testing.assert_close(camera[0,:2]/camera[0,2],torch.tensor([.2,-.1],dtype=torch.float64),atol=1e-7,rtol=1e-6)
        scaled={k:v.clone() for k,v in raw.items()};scaled['xyz']*=10;scaled['logscale']+=math.log(10)
        inv2=inverse.clone();inv2[3,:3]*=10;cam2=view(matrix=torch.linalg.inv(inv2))
        b,_,_=residual_support_seed(scaled,torch.tensor([0]),[evidence(cam2)],1e-5)
        torch.testing.assert_close(b['xyz'],a['xyz']*10)
        torch.testing.assert_close(b['logscale'].exp(),a['logscale'].exp()*10)
    def test_view_supported_medoid_is_deterministic_and_not_mid_air_average(self):
        raw=self.raw();a=evidence(view('cam02.png'),mean=(.4,0),mass=10)
        b=evidence(view('cam01.png'),mean=(-.4,0),mass=1)
        seed,_,report=residual_support_seed(raw,torch.tensor([0]),[a,b],1e-6)
        repeat,_,_=residual_support_seed(raw,torch.tensor([0]),[b,a],1e-6)
        torch.testing.assert_close(seed['xyz'],repeat['xyz'])
        self.assertGreater(float(seed['xyz'][0,0]),1.99)
        self.assertEqual(report['selected_view_counts'],[0,1])
    def test_zero_or_missing_evidence_and_invalid_sources_are_explicit(self):
        raw=self.raw();item=evidence(view(),mass=0)
        seed,valid,_=residual_support_seed(raw,torch.tensor([0]),[item],1e-6)
        self.assertFalse(valid.any());self.assertTrue(all(torch.isfinite(x).all() for x in seed.values()))
        with self.assertRaises(ValueError):residual_support_seed(raw,torch.tensor([0]),[],1e-6)
        bad=self.raw();bad['rotation'].zero_()
        with self.assertRaisesRegex(ValueError,'quaternion'):residual_support_seed(bad,torch.tensor([0]),[item],1e-6)
        with self.assertRaises(ValueError):residual_support_seed(raw,torch.tensor([0]),[item],0)
    def test_quaternion_conversion_handles_half_turns(self):
        matrices=torch.stack((torch.eye(3),torch.diag(torch.tensor([1.,-1.,-1.])),torch.diag(torch.tensor([-1.,1.,-1.])))).double()
        q=_matrix_quaternion(matrices)
        torch.testing.assert_close(_rotation(q),matrices)
        torch.testing.assert_close(q.norm(dim=-1),torch.ones(3,dtype=torch.float64))
    def test_relocated_support_opacity_uses_error_amplitude_not_copied_alpha(self):
        cache=[dict(support_first_moments=torch.tensor([[.12,0,0],[.12,0,0],[0.,0,0]]),native_mass=torch.ones(3))]
        parent=torch.logit(torch.tensor([[.9],[.05],[.8]]))
        result=residual_support_opacity(parent,torch.arange(3),cache).sigmoid()
        torch.testing.assert_close(result[:,0],torch.tensor([.2,.05,0.]),atol=1e-7,rtol=1e-5)
        self.assertTrue(result.isfinite().all())
        mass,mean,cov=support_distribution(torch.zeros(1,3),torch.zeros(1,3),100,100)
        self.assertTrue(torch.isfinite(mean).all());self.assertTrue(torch.isfinite(cov).all())
    def test_policy_is_opt_in_and_requires_frame_plan(self):
        self.assertFalse(BirthConfig().residual_support_init)
        with self.assertRaisesRegex(ValueError,'frame training plan'):
            BirthConfig(residual_support_init=True).validate(200)


if __name__=='__main__':unittest.main()
