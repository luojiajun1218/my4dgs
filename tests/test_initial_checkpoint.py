"""Static seed pruning, reconstruction and SH upgrade remain trainable."""
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from scripts.compact_initial_model import cpu_tree, coverage_selection
from scripts.rebuild_initial_checkpoint import load_reference, rebuild_packet, verify_packet
from scripts.run_first_frame import (COLOUR_FIELDS, STATIC_FIELDS,
    accumulate_camera_frontier_stats, accumulate_extra_densification_views,
    average_optimizer_gradients, camera_frontier_budgeted_selection,
    configure_static_parameters, coverage_budgeted_densify_gaussians,
    coverage_budgeted_selection, enforce_world_scale_cap,
    install_high_order_only_training, progressive_densification_cap,
    static_field_digests, verify_colour_only)
from scripts.upgrade_initial_sh_degree import upgrade_checkpoint


class InitialCheckpointTest(unittest.TestCase):
    def test_world_scale_cap_projects_axes_and_clears_only_clipped_adam(self):
        scaling = torch.nn.Parameter(torch.tensor([
            [-2., -1., .2], [.5, -3., -4.]], dtype=torch.float32))
        optimizer = torch.optim.Adam([scaling], lr=.01)
        optimizer.state[scaling] = {
            'step': torch.tensor(3.),
            'exp_avg': torch.ones_like(scaling),
            'exp_avg_sq': torch.full_like(scaling, 2.),
        }
        model = SimpleNamespace(_scaling=scaling, optimizer=optimizer)
        result = enforce_world_scale_cap(model, scene_extent=10.,
                                         maximum_extent_fraction=.1)
        self.assertEqual(result['clipped_rows'], 2)
        self.assertEqual(result['clipped_components'], 2)
        torch.testing.assert_close(scaling, torch.tensor([
            [-2., -1., 0.], [0., -3., -4.]]), rtol=0, atol=0)
        clipped = torch.tensor([[False, False, True], [True, False, False]])
        for name in ('exp_avg', 'exp_avg_sq'):
            state = optimizer.state[scaling][name]
            self.assertTrue(state[clipped].eq(0).all())
            self.assertTrue(state[~clipped].ne(0).all())
        for extent, fraction in ((0., .1), (1., 0.), (1., 1.1),
                                 (True, .1), (1., True)):
            with self.subTest(extent=extent, fraction=fraction), self.assertRaises(ValueError):
                enforce_world_scale_cap(model, extent, fraction)

    def test_multiview_parameter_gradients_are_averaged_once(self):
        first = torch.nn.Parameter(torch.tensor([1., 2.]))
        second = torch.nn.Parameter(torch.tensor([3.]))
        first.grad = torch.tensor([6., 10.])
        second.grad = torch.tensor([14.])
        optimizer = SimpleNamespace(param_groups=[
            {'params': [first]}, {'params': [first, second]}])
        average_optimizer_gradients(optimizer, 2)
        torch.testing.assert_close(first.grad, torch.tensor([3., 5.]))
        torch.testing.assert_close(second.grad, torch.tensor([7.]))
        for invalid in (0, -1, True, 1.5):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                average_optimizer_gradients(optimizer, invalid)

    def test_extra_multiview_densification_evidence_does_not_trigger_topology(self):
        class Gaussians:
            def __init__(self):
                self.max_radii2D = torch.tensor([1., 2., 3.])
                self.calls = []
            def add_densification_stats(self, points, visible):
                self.calls.append((points.clone(), visible.clone()))
        module = SimpleNamespace(gaussians=Gaussians())
        packages = []
        for offset in (0., 10., 20.):
            packages.append(dict(
                visibility_filter=torch.tensor([True, False, True]),
                radii=torch.tensor([offset + 2., offset + 9., offset + 4.]),
                viewspace_points=torch.full((3, 3), offset)))
        accumulate_extra_densification_views(module, packages)
        # Package zero is left for the one native densify call. Only the two
        # extra views contribute here, and no topology method is invoked.
        self.assertEqual(len(module.gaussians.calls), 2)
        torch.testing.assert_close(module.gaussians.max_radii2D,
                                   torch.tensor([22., 2., 24.]))
        self.assertTrue(all(call[1].tolist() == [True, False, True]
                            for call in module.gaussians.calls))

    def test_coverage_budget_prefers_supported_error_and_is_strict(self):
        # Row zero has the larger one-view gradient, but row one carries a
        # reproducible error over the full camera sample window.
        result = coverage_budgeted_selection(
            gradient_sum=torch.tensor([[.4], [10.], [8.], [8.]]),
            observations=torch.tensor([[1.], [100.], [100.], [100.]]),
            sample_count=100,
            scaling=torch.tensor([[.01,.01,.01], [.01,.01,.01],
                                  [.2,.2,.2], [.2,.2,.2]]),
            percent_dense=.1, scene_extent=1., grad_threshold=.02,
            remaining=2)
        self.assertEqual(result['selected'].nonzero(as_tuple=True)[0].tolist(),[1,2])
        self.assertEqual(result['clone'].nonzero(as_tuple=True)[0].tolist(),[1])
        self.assertEqual(result['split'].nonzero(as_tuple=True)[0].tolist(),[2])
        self.assertAlmostEqual(float(result['score'][0]),.04,places=6)
        self.assertAlmostEqual(float(result['score'][1]),.1,places=6)

    def test_coverage_budget_ties_are_stable_and_inputs_are_guarded(self):
        kwargs=dict(gradient_sum=torch.ones(4,1),observations=torch.ones(4,1),
            sample_count=1,scaling=torch.ones(4,3),percent_dense=.1,
            scene_extent=1.,grad_threshold=.5,remaining=2)
        result=coverage_budgeted_selection(**kwargs)
        self.assertEqual(result['selected'].nonzero(as_tuple=True)[0].tolist(),[0,1])
        for name,value in (('sample_count',0),('remaining',-1),('remaining',True)):
            invalid=dict(kwargs);invalid[name]=value
            with self.subTest(name=name,value=value),self.assertRaises(ValueError):
                coverage_budgeted_selection(**invalid)

    def test_camera_frontier_quota_protects_each_rare_view(self):
        result=camera_frontier_budgeted_selection(
            gradient_sum=torch.tensor([[5.],[2.],[2.],[.1]]),
            observations=torch.tensor([[2.],[1.],[1.],[1.]]),
            camera_gradient_sum=torch.tensor([[2.5,2.,0.,0.],[2.5,0.,2.,.1]]),
            camera_observations=torch.tensor([[1.,1.,0.,0.],[1.,0.,1.,1.]]),
            scaling=torch.full((4,3),.01),percent_dense=.1,scene_extent=1.,
            grad_threshold=.05,remaining=2,frontier_fraction=1.,sample_count=2)
        # A native global top-k would choose shared row0 then row1. Equal
        # rare-view quotas instead preserve one frontier for each camera.
        self.assertEqual(result['selected'].nonzero(as_tuple=True)[0].tolist(),[1,2])
        self.assertEqual(result['camera_allocations'],[1,1])
        self.assertEqual(int(result['frontier_selected'].sum()),2)
        self.assertEqual(int(result['global_selected'].sum()),0)
        rotated=camera_frontier_budgeted_selection(
            gradient_sum=torch.tensor([[5.],[2.],[2.],[.1]]),
            observations=torch.tensor([[2.],[1.],[1.],[1.]]),
            camera_gradient_sum=torch.tensor([[2.5,2.,0.,0.],[2.5,0.,2.,.1]]),
            camera_observations=torch.tensor([[1.,1.,0.,0.],[1.,0.,1.,1.]]),
            scaling=torch.full((4,3),.01),percent_dense=.1,scene_extent=1.,
            grad_threshold=.05,remaining=1,frontier_fraction=1.,sample_count=2,
            camera_offset=1)
        self.assertEqual(rotated['selected'].nonzero(as_tuple=True)[0].tolist(),[2])
        self.assertEqual(rotated['camera_allocations'],[0,1])

    def test_camera_frontier_cannot_bypass_multiview_coverage_gate(self):
        result=camera_frontier_budgeted_selection(
            gradient_sum=torch.tensor([[4.],[.2],[.2]]),
            observations=torch.tensor([[20.],[1.],[1.]]),
            camera_gradient_sum=torch.tensor([[2.,.2,0.],[2.,0.,.2]]),
            camera_observations=torch.tensor([[10.,1.,0.],[10.,0.,1.]]),
            scaling=torch.full((3,3),.01),percent_dense=.1,scene_extent=1.,
            grad_threshold=.05,remaining=3,frontier_fraction=1.,sample_count=100)
        # Rows one and two exceed the old per-camera threshold, but their
        # one-sample coverage score is only .02.  The safe frontier must leave
        # those slots empty instead of manufacturing unsupported topology.
        self.assertEqual(result['selected'].nonzero(as_tuple=True)[0].tolist(),[0])
        self.assertEqual(result['eligible'].nonzero(as_tuple=True)[0].tolist(),[0])
        self.assertEqual(int(result['selected_below_coverage_gate'].sum()),0)
        self.assertAlmostEqual(float(result['score'][1]),.02,places=6)

    def test_camera_frontier_accumulates_unscaled_per_view_gradients(self):
        module=SimpleNamespace(
            _frontier_camera_names=['cam01.png','cam02.png'],
            _frontier_gradient_sum=torch.zeros(2,3),
            _frontier_observations=torch.zeros(2,3),
            gaussians=SimpleNamespace(get_xyz=torch.zeros(3,3)))
        packages=[]
        for name,gradient,visible in (
                ('cam01.png',torch.tensor([[3.,4.,0.],[9.,0.,0.],[0.,6.,8.]]),
                 torch.tensor([True,False,True])),
                ('cam02.png',torch.tensor([[0.,2.,0.],[5.,12.,0.],[1.,1.,0.]]),
                 torch.tensor([True,True,False]))):
            points=torch.zeros(3,3,requires_grad=True);points.grad=gradient
            packages.append(dict(_static_camera_name=name,viewspace_points=points,
                                 visibility_filter=visible))
        accumulate_camera_frontier_stats(module,packages)
        torch.testing.assert_close(module._frontier_gradient_sum,
            torch.tensor([[5.,0.,6.],[2.,13.,0.]]))
        torch.testing.assert_close(module._frontier_observations,
            torch.tensor([[1.,0.,1.],[1.,1.,0.]]))

    def test_coverage_budgeted_topology_never_crosses_cap(self):
        class Model:
            def __init__(self):
                self._xyz=torch.zeros(4,3)
                self._scaling=torch.tensor([[.01,.01,.01],[.01,.01,.01],
                                            [.2,.2,.2],[.2,.2,.2]])
                self._opacity=torch.ones(4,1)
                self.xyz_gradient_accum=torch.ones(4,1)
                self.denom=torch.ones(4,1)
                self.max_radii2D=torch.zeros(4)
                self.percent_dense=.1
            @property
            def get_xyz(self):return self._xyz
            @property
            def get_scaling(self):return self._scaling
            @property
            def get_opacity(self):return self._opacity
            def add_densification_stats(self,*args):pass
            def _append(self,count,scale):
                self._xyz=torch.cat((self._xyz,torch.zeros(count,3)))
                self._scaling=torch.cat((self._scaling,torch.full((count,3),scale)))
                self._opacity=torch.cat((self._opacity,torch.ones(count,1)))
                self.max_radii2D=torch.zeros(len(self._xyz))
            def densify_and_clone(self,carrier,threshold,extent):
                selected=(carrier.flatten()>=threshold)&(self._scaling[:len(carrier)].max(1).values<=.1)
                self._append(int(selected.sum()),.01)
            def densify_and_split(self,carrier,threshold,extent):
                selected=(carrier.flatten()>=threshold)&(self._scaling[:len(carrier)].max(1).values>.1)
                self._append(int(selected.sum()),.2)  # net +1 after replacing each parent
            def prune_points(self,remove):
                keep=~remove
                self._xyz,self._scaling,self._opacity=(self._xyz[keep],self._scaling[keep],self._opacity[keep])
                self.max_radii2D=self.max_radii2D[keep]
            def reset_opacity(self):raise AssertionError('unexpected reset')
        model=Model()
        module=SimpleNamespace(gaussians=model,
            config=SimpleNamespace(densify=SimpleNamespace(from_step=0,interval=1,
                max_gaussians=6,grad_threshold=.01,opacity_reset_interval=100)),
            datamodule=SimpleNamespace(cameras_extent=1.,
                config=SimpleNamespace(white_background=False)))
        event,report=coverage_budgeted_densify_gaussians(module,1,
            torch.ones(4,dtype=torch.bool),torch.zeros(4,3),torch.ones(4),1)
        self.assertTrue(event)
        self.assertEqual(len(model.get_xyz),6)
        self.assertEqual(report['selected_count'],2)
        self.assertEqual(report['post_densify_gaussians'],6)

    def test_progressive_cap_covers_each_real_event_and_reaches_final(self):
        caps=[progressive_densification_cap(step,500,5000,100,40000,200000)
              for step in range(600,5000,100)]
        self.assertEqual(len(caps),44)
        self.assertGreater(caps[0],40000)
        self.assertEqual(caps[-1],200000)
        self.assertTrue(all(left < right for left,right in zip(caps,caps[1:])))
        for invalid in (500,550,5000):
            with self.subTest(step=invalid),self.assertRaises(ValueError):
                progressive_densification_cap(invalid,500,5000,100,40000,200000)

    def test_coverage_keeps_best_available_points_in_every_region(self):
        scores = torch.tensor([9., 8., 7., 6., 5., 4., .1, .0])
        groups = torch.tensor([0, 0, 0, 0, 1, 1, 2, 2])
        selected = coverage_selection(scores, groups, 6)
        self.assertEqual(selected.tolist(), [0, 1, 4, 5, 6, 7])
        self.assertEqual(torch.bincount(groups[selected]).tolist(), [2, 2, 2])

    def test_coverage_ties_are_stable_and_remaining_budget_uses_global_rank(self):
        scores = torch.ones(7)
        groups = torch.tensor([2, 0, 1, 0, 1, 0, 1])
        selected = coverage_selection(scores, groups, 6)
        self.assertEqual(selected.tolist(), [0, 1, 2, 3, 4, 5])
        with self.assertRaisesRegex(ValueError, 'requires'):
            coverage_selection(scores, groups, 4)

    def make_model(self, *, parameters=True):
        model = SimpleNamespace()
        for name in STATIC_FIELDS:
            value = torch.tensor([0.2, -0.4])
            setattr(model, name, torch.nn.Parameter(value) if parameters else value)
        model.optimizer = torch.optim.Adam([getattr(model, name) for name in STATIC_FIELDS], lr=0.01)
        return model

    def test_colour_only_freezes_geometry_even_with_existing_adam_moments(self):
        model = self.make_model()
        sum(getattr(model, name).square().sum() for name in STATIC_FIELDS).backward()
        model.optimizer.step()
        before = static_field_digests(model)
        configure_static_parameters(model, colour_only=True)
        self.assertTrue(all(getattr(model, name).grad is None for name in STATIC_FIELDS))
        sum((getattr(model, name)-1).square().sum() for name in STATIC_FIELDS).backward()
        model.optimizer.step()
        after = static_field_digests(model)
        self.assertEqual(set(verify_colour_only(before, after)), set(COLOUR_FIELDS))

    def test_old_detached_seed_can_resume_native_full_fitting(self):
        model = self.make_model(parameters=False)
        configure_static_parameters(model, colour_only=False)
        before = static_field_digests(model)
        sum(getattr(model, name).square().sum() for name in STATIC_FIELDS).backward()
        model.optimizer.step()
        after = static_field_digests(model)
        self.assertTrue(all(before[name] != after[name] for name in STATIC_FIELDS))

    def test_colour_only_verifier_rejects_geometry_change_and_no_fit(self):
        model = self.make_model()
        before = static_field_digests(model)
        with self.assertRaisesRegex(ValueError, 'did not change'):
            verify_colour_only(before, before)
        changed = dict(before, _xyz='changed')
        with self.assertRaisesRegex(ValueError, 'frozen attributes'):
            verify_colour_only(before, changed)

    def test_high_order_only_masks_low_order_values_and_adam_momentum(self):
        model = self.make_model()
        model._features_dc = torch.nn.Parameter(torch.randn(2, 1, 3))
        model._features_rest = torch.nn.Parameter(torch.randn(2, 15, 3))
        model.optimizer = torch.optim.Adam([
            {'params': [model._xyz], 'name': 'xyz'},
            {'params': [model._features_dc], 'name': 'f_dc'},
            {'params': [model._features_rest], 'name': 'f_rest'},
            {'params': [model._opacity], 'name': 'opacity'},
            {'params': [model._scaling], 'name': 'scaling'},
            {'params': [model._rotation], 'name': 'rotation'},
        ], lr=0.01)
        sum(getattr(model, name).square().sum() for name in STATIC_FIELDS).backward()
        model.optimizer.step()
        model.optimizer.zero_grad(set_to_none=True)

        configure_static_parameters(model, colour_only=True, preserve_sh_degree=1)
        hook, preserved = install_high_order_only_training(model, 1)
        dc_before = model._features_dc.detach().clone()
        low_before = model._features_rest[:, :preserved].detach().clone()
        high_before = model._features_rest[:, preserved:].detach().clone()
        model._features_rest.square().sum().backward()
        model.optimizer.step()
        hook.remove()

        torch.testing.assert_close(model._features_dc, dc_before, rtol=0, atol=0)
        torch.testing.assert_close(model._features_rest[:, :preserved], low_before,
                                   rtol=0, atol=0)
        self.assertFalse(torch.equal(model._features_rest[:, preserved:], high_before))
        state = model.optimizer.state[model._features_rest]
        self.assertEqual(int(torch.count_nonzero(state['exp_avg'][:, :preserved])), 0)
        self.assertEqual(int(torch.count_nonzero(state['exp_avg_sq'][:, :preserved])), 0)

    def test_saved_parameter_keeps_adam_resume_behavior(self):
        parameter = torch.nn.Parameter(torch.tensor([0.2, -0.4]))
        optimizer = torch.optim.Adam([parameter], lr=0.01)

        def step(value, opt):
            (value - torch.tensor([0.7, 0.1])).square().sum().backward()
            opt.step()
            opt.zero_grad(set_to_none=True)

        step(parameter, optimizer)
        buffer = io.BytesIO()
        torch.save(cpu_tree(dict(parameter=parameter, optimizer=optimizer.state_dict())), buffer)
        buffer.seek(0)
        restored = torch.load(buffer, weights_only=False)
        resumed = restored['parameter']
        self.assertIsInstance(resumed, torch.nn.Parameter)
        self.assertTrue(resumed.requires_grad)
        self.assertTrue(resumed.is_leaf)
        self.assertTrue(torch.equal(parameter, resumed))
        resumed_optimizer = torch.optim.Adam([resumed], lr=0.01)
        resumed_optimizer.load_state_dict(restored['optimizer'])
        step(parameter, optimizer)
        step(resumed, resumed_optimizer)
        self.assertTrue(torch.equal(parameter, resumed))

    def test_frozen_parameter_and_state_tensors_stay_frozen(self):
        frozen = torch.nn.Parameter(torch.tensor([1.0]), requires_grad=False)
        state = torch.tensor([2.0], requires_grad=True)
        copied = cpu_tree((frozen, [state]))
        self.assertIsInstance(copied[0], torch.nn.Parameter)
        self.assertFalse(copied[0].requires_grad)
        self.assertFalse(copied[1][0].requires_grad)
        self.assertTrue(torch.equal(copied[0], frozen))
        self.assertTrue(torch.equal(copied[1][0], state))


class RebuildInitialCheckpointTest(unittest.TestCase):
    def make_template(self, count=2):
        shapes = ((count, 3), (count, 1, 3), (count, 3, 3),
                  (count, 3), (count, 4), (count, 1))
        parameters = [torch.nn.Parameter(torch.zeros(shape)) for shape in shapes]
        groups = [{'params': [index], 'name': name, 'lr': 0.1}
                  for index, name in enumerate(
                      ('xyz', 'f_dc', 'f_rest', 'opacity', 'scaling', 'rotation'))]
        optimizer = {'state': {0: {'exp_avg': torch.ones(count, 3)}},
                     'param_groups': groups}
        capture = (1, *parameters, torch.zeros(count), torch.zeros(count, 1),
                   torch.zeros(count, 1), optimizer, 2.0)
        return {'global_step': 11, 'current_frame': 0, 'gaussians': capture,
                'deformation': {}}

    def test_rebuild_is_exact_and_clears_row_dependent_state(self):
        count = 4
        raw = {
            'xyz': torch.arange(count * 3, dtype=torch.float32).reshape(count, 3),
            'sh_dc': torch.ones(count, 1, 3),
            'sh_rest': torch.ones(count, 3, 3) * 2,
            'logscale': torch.ones(count, 3) * 3,
            'rotation': torch.ones(count, 4) * 4,
            'opacity': torch.ones(count, 1) * 5,
        }
        packet = rebuild_packet(self.make_template(), raw, 6.5, 1, 10000)
        verify_packet(packet, raw, 6.5, 1, 10000)
        self.assertEqual(packet['gaussians'][10]['state'], {})
        self.assertEqual(packet['gaussians'][7].shape, (count,))

    def test_reference_validation_and_degree_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'reference.npz'
            count = 3
            np.savez(path, xyz=np.zeros((count, 3), np.float32),
                     sh_dc=np.zeros((count, 1, 3), np.float32),
                     sh_rest=np.zeros((count, 8, 3), np.float32),
                     logscale=np.zeros((count, 3), np.float32),
                     rotation=np.zeros((count, 4), np.float32),
                     opacity=np.zeros((count, 1), np.float32),
                     extent=np.float64(4.0), frame_index=np.int32(0))
            raw, extent, degree = load_reference(path)
            self.assertEqual((len(raw['xyz']), extent, degree), (count, 4.0, 2))


class UpgradeInitialSHDegreeTest(unittest.TestCase):
    def checkpoint(self):
        n = 4
        rest = torch.arange(n * 3 * 3, dtype=torch.float32).reshape(n, 3, 3)
        optimizer = {
            'state': {
                2: {
                    'step': torch.tensor(7.0),
                    'exp_avg': rest + 100,
                    'exp_avg_sq': rest + 200,
                },
            },
            'param_groups': [
                {'name': 'xyz', 'params': [0]},
                {'name': 'f_rest', 'params': [2]},
            ],
        }
        gaussians = (
            1,
            torch.randn(n, 3),
            torch.randn(n, 1, 3),
            rest,
            torch.randn(n, 3),
            torch.randn(n, 4),
            torch.randn(n, 1),
            torch.randn(n),
            torch.randn(n, 1),
            torch.randn(n, 1),
            optimizer,
            2.5,
        )
        return {'global_step': 11_000, 'current_frame': 0,
                'gaussians': gaussians, 'deformation': {'keep': 3}}

    def test_expands_coefficients_and_moments_without_changing_source(self):
        source = self.checkpoint()
        source_rest = source['gaussians'][3].clone()
        upgraded, report = upgrade_checkpoint(source, 3)

        self.assertEqual(upgraded['gaussians'][0], 3)
        self.assertEqual(tuple(upgraded['gaussians'][3].shape), (4, 15, 3))
        torch.testing.assert_close(upgraded['gaussians'][3][:, :3], source_rest,
                                   rtol=0, atol=0)
        self.assertEqual(int(torch.count_nonzero(upgraded['gaussians'][3][:, 3:])), 0)
        self.assertEqual(tuple(source['gaussians'][3].shape), (4, 3, 3))
        for key in ('exp_avg', 'exp_avg_sq'):
            old = source['gaussians'][10]['state'][2][key]
            new = upgraded['gaussians'][10]['state'][2][key]
            torch.testing.assert_close(new[:, :3], old, rtol=0, atol=0)
            self.assertEqual(int(torch.count_nonzero(new[:, 3:])), 0)
        self.assertEqual(report['target_degree'], 3)
        self.assertTrue(report['non_sh_fields_bit_exact'])

    def test_rejects_non_increasing_degree(self):
        with self.assertRaisesRegex(ValueError, 'must exceed'):
            upgrade_checkpoint(self.checkpoint(), 1)


if __name__ == "__main__":
    unittest.main()
