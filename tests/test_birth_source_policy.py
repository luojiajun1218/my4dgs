"""Oriented covariance births, source selection, and optional fixed-support refits."""
from types import SimpleNamespace
import math
import unittest
from unittest.mock import patch

import torch

from dataclasses import asdict

from pipeline.module import BIRTH_ALLOCATION, BirthConfig, Module, ModuleConfig
from pipeline.model.birth_quantization import RAW_ATTRIBUTES, RAW_FIELDS, QuantizedNewGaussianView, model_raw
from pipeline.model.gaussian import BirthGaussianModel, GaussianModel
from pipeline.model.multiview_birth import MultiViewBirthConfig, initial_alpha, temporal_change_scores
from _motion_fixtures import current_motion


class CovarianceBirthTests(unittest.TestCase):
    @staticmethod
    def source():
        source = GaussianModel(1)
        raw = dict(xyz=torch.zeros(1, 3),
            rotation=torch.tensor([[2 ** -.5, 0., 0., 2 ** -.5]]),
            logscale=torch.tensor([[.1, .01, .001]]).log(),
            opacity=torch.tensor([[2.]]), sh_dc=torch.ones(1, 1, 3),
            sh_rest=torch.zeros(1, 3, 3))
        for name, attr in zip(RAW_FIELDS, RAW_ATTRIBUTES):
            setattr(source, attr, torch.nn.Parameter(raw[name].clone()))
        source.active_sh_degree = 1
        source.spatial_lr_scale = 2.
        return source

    def test_oriented_sampling_inherits_attributes_and_true_alpha_without_editing_old(self):
        source = self.source()
        before = model_raw(source)
        births = BirthGaussianModel(1)
        with patch.object(births, 'training_setup'), patch('torch.normal',
                return_value=torch.tensor([[.2, 0., 0.]])) as normal:
            births.initialize_covariance_sources(source, torch.tensor([0]),
                                                  180, object(), sigma_scale=2.)
        torch.testing.assert_close(births._xyz, torch.tensor([[0., .2, 0.]]),
                                   rtol=0, atol=1e-7)
        torch.testing.assert_close(normal.call_args.args[1],
                                   source.get_scaling * 2)
        torch.testing.assert_close(births._opacity, torch.logit(torch.tensor([[.1]])))
        self.assertAlmostEqual(float(births.get_opacity), .1, places=6)
        for name in ('rotation', 'logscale', 'sh_dc', 'sh_rest'):
            torch.testing.assert_close(model_raw(births)[name], before[name], rtol=0, atol=0)
        for name, value in model_raw(source).items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)
        self.assertEqual(births.source_rows.tolist(), [0])
        self.assertEqual(births.created_steps.tolist(), [180])
        self.assertEqual(births.spatial_lr_scale, 2.)

    def test_empty_selection_has_no_optimizer_and_keeps_aligned_empty_attributes(self):
        births = BirthGaussianModel(1)
        births.initialize_covariance_sources(self.source(), torch.empty(0, dtype=torch.long),
                                              180, object())
        self.assertIsNone(births.optimizer)
        self.assertEqual(len(births.source_rows), 0)
        self.assertTrue(all(len(value) == 0 for value in model_raw(births).values()))

    def test_quantized_alpha_matches_requested_coverage_and_keeps_gradient(self):
        source = self.source()
        for initial in (.1, .35):
            with self.subTest(initial_alpha=initial):
                births = BirthGaussianModel(1)
                with patch.object(births, 'training_setup'):
                    births.initialize_covariance_sources(source, torch.tensor([0]),
                        180, object(), initial_alpha=initial)
                view = QuantizedNewGaussianView(births, BirthConfig(), 2.,
                                               model_raw(source), births.source_rows)
                self.assertAlmostEqual(float(view.get_opacity), initial, delta=.002)
                view.get_opacity.sum().backward()
                self.assertGreater(float(births._opacity.grad.abs().sum()), .05)

    def test_initial_alpha_rejects_invalid_configuration(self):
        for value in (0., 1., -.1, float('nan'), float('inf'), True):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'initial_alpha'):
                BirthConfig(initial_alpha=value).validate(180)

    def test_large_sources_seed_finer_children_without_changing_parent(self):
        source=self.source();before=model_raw(source);births=BirthGaussianModel(1)
        with patch.object(births,'training_setup'):
            births.initialize_covariance_sources(source,torch.tensor([0]),180,object(),
                                                  split_scale_threshold=.05)
        torch.testing.assert_close(births.get_scaling,source.get_scaling/1.6)
        torch.testing.assert_close(model_raw(source)['logscale'],before['logscale'])
        torch.testing.assert_close(births._rotation,source._rotation)

    def test_screen_bound_shrinks_covariance_and_jitter_together(self):
        source=self.source();source.max_radii2D=torch.tensor([100.])
        before=model_raw(source);births=BirthGaussianModel(1)
        with patch.object(births,'training_setup'),patch('torch.normal',
                return_value=torch.zeros(1,3)) as normal:
            births.initialize_covariance_sources(source,torch.tensor([0]),180,object(),
                sigma_scale=.5,split_scale_threshold=.05,max_screen_size=20.)
        torch.testing.assert_close(births.get_scaling,source.get_scaling*.1)
        torch.testing.assert_close(normal.call_args.args[1],births.get_scaling*.5)
        for name,value in before.items():
            torch.testing.assert_close(model_raw(source)[name],value,rtol=0,atol=0)

    def test_previous_birth_initializations_are_replay_only(self):
        harness = SimpleNamespace(
            gaussians=SimpleNamespace(restore=lambda *args: None),
            renderer=SimpleNamespace(), config=ModuleConfig(), current_frame=1,
            birth_allocation=BIRTH_ALLOCATION)
        for allocation in ('iterative_residual_proposal_v3', 'covariance_split_balanced_v4',
                           'covariance_soft_birth_v5', 'covariance_refine_budget_v6',
                           'covariance_screen_budget_v7'):
            with self.subTest(allocation=allocation), self.assertRaisesRegex(
                    ValueError, 'newborn allocation.*replay-only'):
                Module.on_load_checkpoint(harness, dict(gaussians={}, rigid_tree_version=1,
                    current_frame=1, birth_allocation=allocation))


class ResidualProposalTests(unittest.TestCase):
    def test_initial_alpha_tracks_residual_strength(self):
        config = MultiViewBirthConfig(alpha_min=.2, alpha_max=.85)
        ratio = torch.tensor([[1., 3.], [1., 3.]])
        weights = torch.ones_like(ratio)
        support = torch.tensor([2., 2.])
        visible = torch.tensor([2., 2.])

        alpha = initial_alpha(ratio, weights, support, visible, config)

        self.assertAlmostEqual(float(alpha[0]), .2, places=5)
        self.assertAlmostEqual(float(alpha[1]), .2 + .65 * (1 - math.exp(-2.)),
                               places=5)
        self.assertGreater(float(alpha[1]), float(alpha[0]))

    def test_initial_alpha_is_limited_by_multiview_support(self):
        config = MultiViewBirthConfig(alpha_min=.2, alpha_max=.85)
        ratio = torch.tensor([[3.], [3.]])
        support = torch.tensor([1.])
        visible = torch.tensor([2.])

        partial = initial_alpha(ratio, torch.tensor([[1.], [0.]]),
                                support, visible, config)
        full = initial_alpha(ratio, torch.ones_like(ratio), support + 1,
                             visible, config)

        self.assertAlmostEqual(float(partial), .2 + .65 * .5 * (1 - math.exp(-2.)),
                               places=5)
        self.assertGreater(float(full), float(partial))

    def test_proposal_initialization_uses_given_points_and_lineage(self):
        born = dict(
            xyz=torch.tensor([[1., 2., 3.]]),
            rotation=torch.tensor([[1., 0., 0., 0.]]),
            logscale=torch.full((1, 3), -4.),
            opacity=torch.tensor([[0.]]),
            sh_dc=torch.zeros(1, 1, 3),
            sh_rest=torch.zeros(1, 3, 3))
        births = BirthGaussianModel(1)
        births.active_sh_degree = 1
        births.spatial_lr_scale = 1.
        with patch.object(births, 'training_setup'):
            births.initialize_from_sources(None, None, 180, object(),
                                           proposal=(born, torch.tensor([7])))

        torch.testing.assert_close(births._xyz, born['xyz'])
        torch.testing.assert_close(births._opacity, born['opacity'])
        self.assertEqual(births.source_rows.tolist(), [7])
        self.assertEqual(births.created_steps.tolist(), [180])


class OldSourcePolicyTests(unittest.TestCase):
    def test_old_sources_are_observed_small_and_not_already_active(self):
        config = SimpleNamespace(
            birth=BirthConfig(densify_grad_threshold=1e-4),
            gaussian_stage2=SimpleNamespace(percent_dense=.01))
        gaussians = SimpleNamespace(
            denom=torch.ones(4, 1),
            xyz_gradient_accum=torch.full((4, 1), 1e-3),
            get_scaling=torch.tensor([[.005, .005, .005], [.02, .02, .02],
                                      [.005, .005, .005], [.009, .008, .007]]))
        module = SimpleNamespace(config=config, gaussians=gaussians,
            datamodule=SimpleNamespace(cameras_extent=1.),
            births=SimpleNamespace(source_rows=torch.tensor([2])),
            _birth_temporal_change_scores=lambda rows:torch.ones(len(rows)))

        rows = Module._old_birth_sources(module)

        self.assertEqual(rows.tolist(), [0, 3])
        diagnostics = module._old_birth_source_diagnostics
        self.assertEqual({key: value for key, value in diagnostics.items()
                          if key != 'selected_gradient_mean'}, dict(
            observed=4, high_gradient_observed=4, small_high_gradient=3,
            large_high_gradient=1, duplicate_active_source=1, oversized_child_sources=0,
            budget_blocked=0, selected=2, selected_small=2, selected_large=0))
        self.assertAlmostEqual(diagnostics['selected_gradient_mean'], 1e-3)

    def test_budget_prefers_strong_gradients_among_small_sources(self):
        config=SimpleNamespace(birth=BirthConfig(densify_grad_threshold=1.,max_new_gaussians=2),
                               gaussian_stage2=SimpleNamespace(percent_dense=.01))
        gaussians=SimpleNamespace(denom=torch.ones(5,1),
            xyz_gradient_accum=torch.tensor([[3.],[5.],[4.],[6.],[100.]]),
            get_scaling=torch.tensor([[.005],[.007],[.009],[.02],[.5]]).expand(-1,3))
        module=SimpleNamespace(config=config,gaussians=gaussians,
            datamodule=SimpleNamespace(cameras_extent=1.),births=None,
            _birth_temporal_change_scores=lambda rows:torch.ones(len(rows)))
        rows=Module._old_birth_sources(module)
        self.assertEqual(rows.tolist(),[1,2])
        self.assertEqual(module._old_birth_source_diagnostics['selected_small'],2)
        self.assertEqual(module._old_birth_source_diagnostics['selected_large'],0)
        self.assertEqual(module._old_birth_source_diagnostics['oversized_child_sources'],1)
        self.assertEqual(module._old_birth_source_diagnostics['budget_blocked'],1)

    def test_small_source_pool_uses_gradient_without_temporal_rendering(self):
        gaussians=SimpleNamespace(denom=torch.ones(3,1),
            xyz_gradient_accum=torch.tensor([[10.],[2.],[1.]]),
            get_scaling=torch.full((3,3),.005))
        def unexpected_residual_render(rows):
            raise AssertionError('the active birth policy must not render residual maps')

        module=SimpleNamespace(config=SimpleNamespace(birth=BirthConfig(
            densify_grad_threshold=.5,max_new_gaussians=1),
            gaussian_stage2=SimpleNamespace(percent_dense=.01)),gaussians=gaussians,
            datamodule=SimpleNamespace(cameras_extent=1.),births=None,
            _birth_temporal_change_scores=unexpected_residual_render)
        self.assertEqual(Module._old_birth_sources(module).tolist(),[0])

    def test_temporal_pixel_change_uses_matched_views_and_ignores_invalid_projections(self):
        previous=SimpleNamespace(image=torch.zeros(3,5,5),full_proj_transform=torch.eye(4))
        current=SimpleNamespace(image=previous.image.clone(),full_proj_transform=torch.eye(4),
                                world_view_transform=torch.eye(4))
        current.image[:,2,1]=.3
        xyz=torch.tensor([[-.4,0.,1.],[.4,0.,1.],[0.,0.,-1.],[2.,0.,1.]])
        scores=temporal_change_scores(xyz,[(current,previous)])
        torch.testing.assert_close(scores,torch.tensor([.3,0.,0.,0.]))
        with self.assertRaisesRegex(ValueError,'camera pairs'):
            temporal_change_scores(xyz,[])
        previous.full_proj_transform=2*torch.eye(4)
        with self.assertRaisesRegex(ValueError,'fixed training cameras'):
            temporal_change_scores(xyz,[(current,previous)])

    def test_candidate_budget_validation(self):
        for budget in (0,-1,True,1.5):
            with self.subTest(budget=budget),self.assertRaisesRegex(ValueError,'max_new_gaussians'):
                BirthConfig(max_new_gaussians=budget).validate(180)

    def test_missing_sigma_in_same_allocation_checkpoint_config_uses_default(self):
        config = ModuleConfig()
        saved_config = asdict(config)
        saved_config['birth'].pop('clone_sigma_scale')
        harness = SimpleNamespace(
            gaussians=SimpleNamespace(restore=lambda packet, cfg: None),
            renderer=SimpleNamespace(), config=config, current_frame=1,
            birth_allocation=BIRTH_ALLOCATION,
            _freeze_base=lambda: None,
            trainer=SimpleNamespace(global_step=300,
                                    config=SimpleNamespace(num_init_steps=15000)))
        checkpoint = dict(
            global_step=300, current_frame=1, rigid_tree_version=1,
            birth_allocation=BIRTH_ALLOCATION,
            gaussians={}, rigid_tree_config=asdict(config.rigid_tree),
            rigid_tree_module_config=saved_config,
            rigid_tree_frame=dict(prepared=False, finalized=True,
                motion_committed=False, reference_count=10, report=None,
                birth_control={}, stage_image_metrics=[]),
            rigid_tree_initial_reference=None, rigid_tree_retired_reference=None,
            births=None)

        Module.on_load_checkpoint(harness, checkpoint)

        self.assertEqual(config.birth.clone_sigma_scale, .5)
        self.assertEqual(harness.reference_count, 10)

    def test_old_source_alpha_cap_uses_true_alpha_and_keeps_gradient(self):
        raw = dict(
            xyz=torch.tensor([[0., 0., 0.]]),
            rotation=torch.tensor([[1., 0., 0., 0.]]),
            logscale=torch.full((1, 3), -5.),
            opacity=torch.tensor([[10.]]),
            sh_dc=torch.zeros(1, 1, 3),
            sh_rest=torch.zeros(1, 3, 3))
        source = GaussianModel(1)
        for name, attr in zip(RAW_FIELDS, RAW_ATTRIBUTES):
            setattr(source, attr, torch.nn.Parameter(raw[name].clone()))
        source.active_sh_degree = 1
        source.spatial_lr_scale = 1.
        births = BirthGaussianModel(1)
        with patch.object(births, 'training_setup'):
            births.initialize_from_sources(source, torch.tensor([0]), 180,
                object(), opacity_alpha_cap=.1)
        reference = model_raw(source)
        view = QuantizedNewGaussianView(births, BirthConfig(), 2., reference,
                                        births.source_rows)

        alpha = view.get_opacity
        torch.testing.assert_close(torch.sigmoid(births._opacity),
                                   torch.tensor([[.1]]), atol=1e-6, rtol=0)
        self.assertTrue(bool((alpha > .09).all() & (alpha < .11).all()))
        alpha.sum().backward()
        self.assertIsNotNone(births._opacity.grad)
        self.assertGreater(float(births._opacity.grad.abs().sum()), .01)

    def test_module_training_step_charges_fixed_support_refit_regularizer(self):
        class FakeModule:
            motion_committed = True
            birth_refit_active = True
            frame_prepared = True
            motion_steps = 2
            config = SimpleNamespace(lambda_dssim=.2, motion_coarse_to_fine=False,
                effective_motion_observations=False,
                rigid_tree=SimpleNamespace(spatial_gate=SimpleNamespace(start_step=0)))
            gaussians = SimpleNamespace(_xyz=torch.zeros(3, 3))
            births = SimpleNamespace(_xyz=torch.zeros(1, 3))
            opacity_update = None
            motion = SimpleNamespace(
                fixed_support_regularization_loss=lambda step: torch.tensor(2.))

            def forward(self, viewpoint, training=False):
                return dict(render=torch.zeros_like(viewpoint.image))

        viewpoint = SimpleNamespace(image=torch.zeros(3, 2, 2))
        module = FakeModule()

        loss, metrics, _ = Module.training_step(module, viewpoint, 3,
                                                  regularization_weight=4.)

        self.assertEqual(float(loss), 8.)
        self.assertEqual(float(metrics['motion_refit_regularization']), 8.)


class FixedSupportRateTests(unittest.TestCase):
    def test_fixed_support_refit_reuses_rate_and_relative_terms_only(self):
        model = current_motion(points_per_axis=4, max_depth=2,
                               amplitude_refit_steps=30)
        model._hard_gate_frozen = True
        model._candidate_initialized = True
        with torch.no_grad():
            model.region_translation[model.owner[0], 0] = 3 * model.steps[0]
            model.gs_translation[0, 0] = 2 * model.gs_steps[0]
        model(model.config.motion_steps - model.config.amplitude_refit_steps + 1,
              training=True)
        for name in ('region_translation', 'region_rotation',
                     'gs_translation', 'gs_rotation'):
            getattr(model, name).requires_grad_(True)

        self.assertTrue(model.amplitude_refit_active)
        self.assertEqual(float(model.regularization_loss(model.last_step)), 0.)
        terms = model.fixed_support_regularization_terms(model.last_step)
        self.assertTrue(model.config.proximal_gs_regularization)
        self.assertTrue(model.config.spatial_gate.shared_image_recode)
        self.assertGreater(float(terms['rate_loss']), 0.)
        self.assertTrue(terms['rate_loss'].requires_grad)
        self.assertGreater(float(terms['relative_motion_loss']), 0.)
        terms['rate_loss'].backward(retain_graph=True)
        active = model._amplitude_refit_support['gs_translation']
        self.assertGreater(float(model.gs_translation.grad[active].abs().sum()), 0.)
        self.assertEqual(float(model.gs_translation.grad[~active].abs().sum()), 0.)
        self.assertTrue(model.region_translation.grad is not None)
        for name in ('region_translation', 'region_rotation',
                     'gs_translation', 'gs_rotation'):
            getattr(model, name).grad = None
        terms['relative_motion_loss'].backward()
        self.assertGreater(float(model.gs_translation.grad[active].abs().sum()), 0.)

    def test_fixed_support_restores_loss_switches_even_when_loss_terms_raises(self):
        model = current_motion(points_per_axis=4, max_depth=2,
                               amplitude_refit_steps=30)
        model._hard_gate_frozen = True
        model._candidate_initialized = True
        model(model.config.motion_steps - model.config.amplitude_refit_steps + 1,
              training=True)
        self.assertTrue(model.config.proximal_gs_regularization)
        self.assertTrue(model.config.spatial_gate.shared_image_recode)

        with patch.object(model, 'loss_terms', side_effect=RuntimeError('diagnostic failure')):
            with self.assertRaisesRegex(RuntimeError, 'diagnostic failure'):
                model.fixed_support_regularization_terms(model.last_step)

        self.assertTrue(model.config.proximal_gs_regularization)
        self.assertTrue(model.config.spatial_gate.shared_image_recode)


if __name__ == '__main__':
    unittest.main()
