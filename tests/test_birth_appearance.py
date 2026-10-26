"""Birth Appearance tests."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import importlib
import itertools
import json
import math
import unittest
from types import ModuleType, SimpleNamespace, SimpleNamespace as NS
from unittest.mock import patch

import torch
from torch import nn

from pipeline.model import (
    birth_radiance as br,
    family_radiance as fr,
    residual_birth as rb,
)
from pipeline.model.birth_quantization import (
    QuantizedNewGaussianView,
    RAW_ATTRIBUTES,
    RAW_FIELDS,
    attribute_steps,
    family_quantization_diagnostics,
    materialize_quantized_gaussians,
    model_raw,
    quantized_symbols,
)
from pipeline.model.conservative_family import (
    ConservativeFamilyModel,
    OPTICAL_MASS_MAX,
    install_conservative_families,
    inverse_optical_depth_integral,
    optical_depth_integral,
)
from pipeline.model.gaussian import BirthGaussianModel
from pipeline.model.predictive_colour import (
    PredictiveColourMoments,
    calibrate_predictive_colour,
    solve_predictive_simplex,
)
from pipeline.model.residual_birth import RawView
from pipeline.model.rigid_residual_tree import RigidTreeConfig
from pipeline.model.spatial_gated_motion import SpatialGatedMotion

# ----- test_birth_radiance -----
def raw_points(count):
    return dict(xyz=torch.stack((torch.arange(count).float(), torch.zeros(count),
                                 torch.full((count,), 3.)), -1),
                rotation=torch.tensor([[1., 0., 0., 0.]]).repeat(count, 1),
                logscale=torch.zeros(count, 3), opacity=torch.zeros(count, 1),
                sh_dc=torch.zeros(count, 1, 3), sh_rest=torch.zeros(count, 0, 3))


class TinyGaussians(torch.nn.Module):
    def __init__(self, raw):
        super().__init__()
        for name, attr in zip(RAW_FIELDS, RAW_ATTRIBUTES):
            setattr(self, attr, torch.nn.Parameter(raw[name].clone(), requires_grad=False))
        self.active_sh_degree = self.max_sh_degree = 0


class CompositingRenderer:
    """CPU alpha*T renderer sharing the production SH and override-color rules."""
    collect_contributions = True

    def __init__(self, gaussians):
        self.gaussians = gaussians
        self.last_contributions = object()

    def weights(self, view):
        scene = self.gaussians
        ids = scene.get_xyz[:, 0].round().long()
        alpha = view.coverage[ids] * scene.get_opacity[:, :, None]
        survival = 1 - alpha
        transmission = torch.cat((torch.ones_like(survival[:1]), survival[:-1])).cumprod(0)
        return alpha * transmission, survival.prod(0)

    def __call__(self, view, background, override_color=None):
        self.last_contributions = None
        weights, transmission = self.weights(view)
        color = (self.gaussians.get_features[:, 0] * br.Y00 + .5).clamp_min(0)
        if override_color is not None:
            color = override_color
        image = (weights[:, None] * color[:, :, None, None]).sum(0)
        return {'render': image + background[:, None, None] * transmission}


class TinyModule:
    def __init__(self):
        self.gaussians = TinyGaussians(raw_points(3))
        new = {name: value[2:].clone() for name, value in raw_points(3).items()}
        new['xyz'][0, 0] = 3.
        self.births = TinyGaussians(new)
        # The original-old source index 2 lies outside the survivor subset of
        # length 2. A materialize_quantized_gaussians(subset, view) call fails.
        self.births.source_rows = torch.tensor([2])
        self.birth_transaction = NS(replacements=torch.tensor([0]), fixed_reference=None)
        self.config = NS(sh_degree=0, birth=NS(new_xyz_step_extent=.01,
            new_rotation_step=.01, new_logscale_step=.01, new_opacity_step=.01,
            new_sh_step=.05, rotation_updates=True))
        self.datamodule = NS(cameras_extent=1.)
        self.background = torch.tensor([.03, .04, .05])
        self.renderer = CompositingRenderer(self.gaussians)
        self.motion_committed, self.frame_finalized = True, False
        self.births.optimizer = torch.optim.Adam([self.births._features_dc], lr=.01)

    def make_birth_view(self):
        reference = self.birth_transaction.fixed_reference
        if reference is None:
            reference = model_raw(self.gaussians)
        return QuantizedNewGaussianView(self.births, self.config.birth,
            self.datamodule.cameras_extent, reference, self.births.source_rows)

    def forward(self, view):
        previous = self.renderer.gaussians
        old = self.birth_transaction.fixed_reference
        if old is None:
            old = model_raw(self.gaussians)
            keep = torch.ones(len(old['xyz']), dtype=torch.bool)
            parents = self.birth_transaction.replacements
            keep[parents[parents >= 0]] = False
            old = {name: value[keep] for name, value in old.items()}
        new = self.make_birth_view()
        raw = {name: torch.cat((old[name], new.raw_value(name))) for name in RAW_FIELDS}
        try:
            self.renderer.gaussians = RawView(raw, 0, self.gaussians)
            return self.renderer(view, self.background)
        finally:
            self.renderer.gaussians = previous


def fixture(delta=(.4, -.2, .3), count=3):
    module = TinyModule()
    views = {}
    for index in range(count):
        view = NS(path=f'fit{index}.png', camera_center=torch.zeros(3),
            coverage=torch.tensor([[[.2, .1]], [[.15, .25]], [[.1, .2]],
                                   [[.3 + .05*index, .5 - .04*index]]]))
        scene, _ = br._response_scene(module)
        module.renderer.gaussians = scene
        with torch.no_grad():
            weights, _ = module.renderer.weights(view)
            image = module.renderer(view, module.background)['render']
            view.image = image + weights[-1][None] * br.Y00 * torch.tensor(delta)[:, None, None]
        views[f'fit{index}.png'] = view
    module.renderer.gaussians = module.gaussians
    module.renderer.last_contributions = object()
    return module, views


class AggregationTests(unittest.TestCase):
    def test_consistent_residual_recovers_intrinsic_dc_in_coefficient_units(self):
        h = torch.tensor([1., 2., 5.], dtype=torch.float64)[:, None, None].expand(-1, 2, 3)
        desired = torch.tensor([[.3, -.4, .7], [-.2, .9, 1.1]], dtype=torch.float64)
        delta, stats = br._coherent_dc_delta(h * desired, h)
        torch.testing.assert_close(delta, desired, rtol=1.e-10, atol=1.e-10)
        torch.testing.assert_close(stats['coherence'], torch.ones_like(desired), rtol=1.e-10, atol=1.e-10)
        self.assertEqual(stats['support'].tolist(), [[3, 3, 3], [3, 3, 3]])

    def test_disagreement_softly_shrinks_without_a_per_view_zero_damage_veto(self):
        h = torch.ones(2, 1, 3, dtype=torch.float64)
        d = torch.tensor([[[1., 1., 1.]], [[-1., -.5, 2.]]], dtype=torch.float64)
        delta, stats = br._coherent_dc_delta(h*d, h)
        torch.testing.assert_close(stats['coherence'][0], torch.tensor([0., .1, .9], dtype=torch.float64))
        torch.testing.assert_close(delta[0], torch.tensor([0., .025, 1.35], dtype=torch.float64))

    def test_zero_single_and_nonfinite_responses_do_not_authorize_updates(self):
        h = torch.tensor([[[0., 1., 1.]], [[0., 0., float('nan')]]])
        b = torch.tensor([[[9., .5, .5]], [[9., 9., .5]]])
        delta, stats = br._coherent_dc_delta(b, h)
        self.assertEqual(stats['support'].tolist(), [[0, 1, 1]])
        self.assertEqual(delta.tolist(), [[0., 0., 0.]])
        self.assertTrue(bool(torch.isfinite(stats['coherence']).all()))


class EnvelopeTests(unittest.TestCase):
    def test_low_response_huge_newton_delta_is_ldr_bounded_and_still_improves(self):
        module, views = fixture()
        module.birth_transaction.raw = model_raw(module.births)
        prior = module.birth_transaction.raw['sh_dc'].clone()
        for view in views.values():
            view.coverage[3] *= .01
            scene, _ = br._response_scene(module)
            module.renderer.gaussians = scene
            weights, _ = module.renderer.weights(view)
            baseline = module.renderer(view, module.background)['render']
            view.image = baseline + weights[-1][None] * br.Y00 * torch.tensor([150., -50., 75.])[:, None, None]
            self.assertTrue(bool(((view.image >= 0) & (view.image <= 1)).all()))
        module.renderer.gaussians = module.gaussians
        report = br.calibrate_intrinsic_dc(module, views)
        self.assertTrue(report['line_search']['accepted'])
        self.assertGreater(report['dc_update']['unbounded_abs']['max'], 100.)
        self.assertLess(report['dc_update']['bounded_abs']['max'], 2.)
        self.assertEqual(report['dc_update']['bounded_points'], 1)
        self.assertEqual(report['dc_update']['bounded_rgb_channels'], 3)
        self.assertEqual(report['source_envelope']['low_rgb'], [[0., 0., 0.]])
        self.assertEqual(report['source_envelope']['high_rgb'], [[1., 1., 1.]])
        self.assertEqual(report['source_envelope']['source'], 'transaction.raw_cohort_initial_dc_and_sh_rest')
        colors = module.make_birth_view().raw_value('sh_dc') * br.Y00 + .5
        self.assertTrue(bool(((colors >= 0) & (colors <= 1)).all()))
        torch.testing.assert_close(module.birth_transaction.raw['sh_dc'], prior, atol=0, rtol=0)
        json.dumps(report, allow_nan=False)

    def test_inherited_directional_hdr_and_negative_sh_keep_their_source_envelope(self):
        module, _ = fixture()
        module.config.sh_degree = 1
        module.gaussians.active_sh_degree = module.gaussians.max_sh_degree = 1
        module.gaussians._features_rest = torch.nn.Parameter(torch.zeros(3, 3, 3), requires_grad=False)
        module.births._features_rest = torch.nn.Parameter(torch.zeros(1, 3, 3), requires_grad=False)
        module.births._features_rest.data[0, 2] = torch.tensor([2., 2., 0.])
        module.births._features_dc.data[0, 0] = torch.tensor([2., -2., 0.])
        module.birth_transaction.raw = model_raw(module.births)
        views = [NS(camera_center=torch.tensor([4., 0., 3.])),
                 NS(camera_center=torch.tensor([2., 0., 3.])),
                 NS(camera_center=torch.tensor([3., 0., 2.]))]
        scene, old_count = br._response_scene(module)
        lower, upper, report = br._source_envelope(module, views, scene, old_count)
        self.assertGreater(report['high_rgb'][0][0], 2.)
        self.assertLess(report['low_rgb'][0][1], -1.)
        self.assertTrue(bool((lower <= 0).all() & (upper >= 0).all()))
        # Neither update may expand its already inherited angular extremum.
        self.assertEqual(float(upper[0, 0]), 0.)
        self.assertEqual(float(lower[0, 1]), 0.)
        delta = torch.tensor([[100., -100., 100.]], dtype=torch.float64).maximum(lower).minimum(upper)
        initial = module.births._features_dc.detach().clone()
        candidate = br._box_quantized_candidate(initial, delta[:, None].float(), 1.,
                                                module.make_birth_view(), lower, upper)
        torch.testing.assert_close(candidate[0, 0, :2], initial[0, 0, :2], atol=0, rtol=0)

    def test_quantized_trial_is_projected_inward_after_continuous_box_projection(self):
        module, views = fixture()
        module.births._features_dc.data.fill_(.017)
        scene, old_count = br._response_scene(module)
        lower, upper, report = br._source_envelope(module, list(views.values()), scene, old_count)
        initial = module.births._features_dc.detach().clone()
        view = module.make_birth_view()
        # Straight rounding of this continuous boundary would produce 1.8 DC,
        # whose color is >1; the closest feasible grid point is 1.75 DC.
        naive = initial + upper[:, None].float()
        naive_decoded = view.predictor['sh_dc'] + ((naive-view.predictor['sh_dc'])/view.steps['sh_dc']).round()*view.steps['sh_dc']
        self.assertGreater(float(naive_decoded.max() * br.Y00 + .5), 1.)
        candidate = br._box_quantized_candidate(initial, upper[:, None].float(), 1., view, lower, upper)
        decoded = view.predictor['sh_dc'] + ((candidate-view.predictor['sh_dc'])/view.steps['sh_dc']).round()*view.steps['sh_dc']
        difference = (decoded-view.raw_value('sh_dc')).double()[:, 0]
        self.assertTrue(bool(((difference >= lower) & (difference <= upper)).all()))
        self.assertEqual(report['source'], 'current_quantized_features_fallback_missing_transaction_prior')

    def test_quantized_prior_mismatch_only_minimally_expands_for_zero_feasibility(self):
        module, views = fixture()
        module.births._features_dc.data.fill_(1.79)
        module.birth_transaction.raw = model_raw(module.births)
        scene, old_count = br._response_scene(module)
        lower, upper, report = br._source_envelope(module, list(views.values()), scene, old_count)
        self.assertEqual(report['zero_step_expanded_rgb_channels'], 3)
        self.assertTrue(bool((lower <= 0).all() & (upper >= 0).all()))
        self.assertEqual(upper.tolist(), [[0., 0., 0.]])
        self.assertGreater(report['zero_step_expansion_abs']['max'], 0.)
        self.assertLess(report['zero_step_expansion_abs']['max'], .01)


class CalibrationTests(unittest.TestCase):
    def test_true_compositing_fit_preserves_parameter_optimizer_and_sh_rest(self):
        module, views = fixture()
        module.birth_transaction.raw = model_raw(module.births)
        parameter = module.births._features_dc
        rest = module.births._features_rest
        parameter.grad = torch.full_like(parameter, 7.)
        optimizer = module.births.optimizer
        optimizer.state[parameter]['exp_avg'] = torch.full_like(parameter, .125)
        state = optimizer.state[parameter]['exp_avg'].clone()
        renderer_scene, last = module.renderer.gaussians, module.renderer.last_contributions
        with torch.no_grad():
            report = br.calibrate_intrinsic_dc(module, views)
        self.assertEqual(report['status'], 'accepted')
        self.assertLess(report['after_mean_mse'], report['before_mean_mse'] * .001)
        self.assertEqual(report['support']['per_point_responding_views'], [3])
        self.assertEqual(report['support']['multiview_points'], 1)
        self.assertIs(module.births._features_dc, parameter)
        self.assertIs(optimizer.param_groups[0]['params'][0], parameter)
        self.assertIs(module.births._features_rest, rest)
        self.assertFalse(parameter.requires_grad)
        self.assertFalse(rest.requires_grad)
        torch.testing.assert_close(parameter.grad, torch.full_like(parameter, 7.))
        torch.testing.assert_close(optimizer.state[parameter]['exp_avg'], state)
        torch.testing.assert_close(parameter[:, 0], torch.tensor([[.4, -.2, .3]]), atol=2.e-5, rtol=0)
        self.assertIs(module.renderer.gaussians, renderer_scene)
        self.assertIs(module.renderer.last_contributions, last)
        self.assertTrue(module.renderer.collect_contributions)
        json.dumps(report, allow_nan=False)

    def test_rejected_search_restores_exact_dc_and_keeps_optimizer_identity(self):
        module, views = fixture()
        parameter, initial = module.births._features_dc, module.births._features_dc.clone()
        with patch.object(br, '_errors', return_value=torch.ones(len(views))):
            report = br.calibrate_intrinsic_dc(module, views)
        self.assertFalse(report['line_search']['accepted'])
        torch.testing.assert_close(parameter, initial, atol=0, rtol=0)
        self.assertEqual(report['before_per_view_mse'], report['after_per_view_mse'])
        self.assertIs(module.births._features_dc, parameter)
        self.assertIs(module.births.optimizer.param_groups[0]['params'][0], parameter)

    def test_backtracking_accepts_joint_scene_after_diagonal_overshoot(self):
        module, views = fixture()
        real_aggregate = br._coherent_dc_delta
        def exaggerated(*args, **kwargs):
            delta, stats = real_aggregate(*args, **kwargs)
            return delta*3, stats
        with patch.object(br, '_coherent_dc_delta', side_effect=exaggerated):
            report = br.calibrate_intrinsic_dc(module, views)
        self.assertTrue(report['line_search']['accepted'])
        self.assertEqual(report['line_search']['accepted_scale'], .5)
        self.assertFalse(report['line_search']['trials'][0]['accepted'])

    def test_error_during_search_rolls_back_value_flags_and_renderer_state(self):
        module, views = fixture()
        parameter, initial = module.births._features_dc, module.births._features_dc.clone()
        parameter.requires_grad_(True)
        renderer_scene, last = module.renderer.gaussians, module.renderer.last_contributions
        with patch.object(br, '_errors', side_effect=RuntimeError('renderer failure')):
            with self.assertRaisesRegex(RuntimeError, 'renderer failure'):
                br.calibrate_intrinsic_dc(module, views)
        self.assertIs(module.births._features_dc, parameter)
        self.assertTrue(parameter.requires_grad)
        torch.testing.assert_close(parameter, initial, atol=0, rtol=0)
        self.assertIs(module.renderer.gaussians, renderer_scene)
        self.assertIs(module.renderer.last_contributions, last)

    def test_probe_failure_restores_original_parameter(self):
        module, views = fixture()
        parameter = module.births._features_dc
        with patch.object(module, 'forward', side_effect=RuntimeError('probe failure')):
            with self.assertRaisesRegex(RuntimeError, 'probe failure'):
                br.calibrate_intrinsic_dc(module, views)
        self.assertIs(module.births._features_dc, parameter)
        self.assertFalse(parameter.requires_grad)

    def test_tiny_quantized_zero_update_stops_without_changing_dc(self):
        module, views = fixture(delta=(.001, .001, .001))
        initial = module.births._features_dc.clone()
        with patch.object(br, '_errors', side_effect=AssertionError('unnecessary forward')):
            report = br.calibrate_intrinsic_dc(module, list(views.values()))
        self.assertEqual(report['reason'], 'quantized_update_is_zero')
        self.assertEqual(len(report['line_search']['trials']), 1)
        torch.testing.assert_close(module.births._features_dc, initial, atol=0, rtol=0)

    def test_scene_is_independent_and_exact_fixed_survivors_take_priority(self):
        module, _ = fixture()
        fixed = {name: value[1:].clone() for name, value in model_raw(module.gaussians).items()}
        fixed['sh_dc'].fill_(.1)
        module.birth_transaction.fixed_reference = fixed
        module.births.source_rows = torch.tensor([1])
        scene, count = br._response_scene(module)
        self.assertEqual(count, 2)
        self.assertEqual(scene.get_xyz[:, 0].tolist(), [1., 2., 3.])
        scene.get_features[0].fill_(99.)
        self.assertAlmostEqual(float(fixed['sh_dc'][0, 0, 0]), .1)
        torch.testing.assert_close(module.gaussians._features_dc, torch.zeros_like(module.gaussians._features_dc))

    def test_sh_and_final_output_clamps_remove_channel_response(self):
        module, views = fixture()
        module.births._features_dc.data[0, 0, 0] = -3.
        scene, old_count = br._response_scene(module)
        view = next(iter(views.values()))
        mask = torch.ones_like(view.image, dtype=torch.bool)
        mask[2].fill_(False)
        response = br._view_response(module, view, scene, old_count, mask)
        self.assertEqual(float(response[0, 0]), 0.)
        self.assertGreater(float(response[0, 1]), 0.)
        self.assertEqual(float(response[0, 2]), 0.)

    def test_zero_births_skips_without_accessing_renderer(self):
        report = br.calibrate_intrinsic_dc(NS(births=NS(_xyz=torch.empty(0, 3))), {})
        self.assertEqual(report['reason'], 'zero_births')
        json.dumps(report, allow_nan=False)

    def test_improving_population_may_accept_one_training_camera_tradeoff(self):
        module, views = fixture(delta=(.4, .4, .4))
        first = next(iter(views.values()))
        scene, _ = br._response_scene(module)
        module.renderer.gaussians = scene
        weights, _ = module.renderer.weights(first)
        first.image -= .6 * br.Y00 * weights[-1][None]
        module.renderer.gaussians = module.gaussians
        report = br.calibrate_intrinsic_dc(module, views)
        self.assertTrue(report['line_search']['accepted'])
        self.assertLess(report['after_mean_mse'], report['before_mean_mse'])
        self.assertGreater(report['after_per_view_mse'][0], report['before_per_view_mse'][0])


# ----- test_birth_radiance_integration -----
MODULE = importlib.import_module('pipeline.module')


def integration_fixture():
    raw = dict(xyz=torch.zeros(1, 3), rotation=torch.tensor([[1., 0., 0., 0.]]),
               logscale=torch.zeros(1, 3), opacity=torch.zeros(1, 1),
               sh_dc=torch.full((1, 1, 3), .25), sh_rest=torch.full((1, 3, 3), .125))
    model = BirthGaussianModel(1)
    for name, attribute in zip(RAW_FIELDS, RAW_ATTRIBUTES):
        setattr(model, attribute, torch.nn.Parameter(raw[name].clone()))
    config = MODULE.ModuleConfig(birth=MODULE.BirthConfig(mechanism='residual_transaction', radiance_mode='inherited_intrinsic'))
    return raw, model, config


class RadianceIntegrationTests(unittest.TestCase):
    def test_geometry_adam_cannot_change_frozen_appearance_or_accumulate_its_moments(self):
        raw, model, config = integration_fixture()
        groups = [{'name': name, 'params': [getattr(model, attribute)]}
                  for name, attribute in zip(RAW_FIELDS, RAW_ATTRIBUTES)]
        model.optimizer = torch.optim.Adam(groups, lr=.1)
        dc, angular = model._features_dc, model._features_rest
        MODULE.freeze_birth_radiance(model, config)
        loss = (model._xyz+1).square().sum()+model._features_dc.square().sum()+model._features_rest.square().sum()
        loss.backward(); model.optimizer.step()
        self.assertIs(model._features_dc, dc)
        self.assertIs(model._features_rest, angular)
        self.assertFalse(dc.requires_grad or angular.requires_grad)
        self.assertIsNone(dc.grad); self.assertIsNone(angular.grad)
        self.assertNotIn(dc, model.optimizer.state)
        self.assertNotIn(angular, model.optimizer.state)
        torch.testing.assert_close(dc, raw['sh_dc'], atol=0, rtol=0)
        torch.testing.assert_close(angular, raw['sh_rest'], atol=0, rtol=0)
        self.assertTrue(bool(model._xyz.ne(raw['xyz']).any()))

    def test_split_inherits_parent_radiance_despite_different_observed_pixel_colors(self):
        raw, _, _ = integration_fixture()
        domain = rb.covariance_support(raw, 0)
        colors = torch.tensor([[.9, .1, .1] if point[0] < 0 else [.1, .1, .9]
                               for point in domain['xyz']])
        children, _, _ = rb.split_support(raw, 0, domain,
            dict(valid=torch.ones(9, dtype=torch.bool), error=torch.ones(9), color=colors))
        for child in children:
            torch.testing.assert_close(child['sh_dc'], raw['sh_dc'][0], atol=0, rtol=0)
            torch.testing.assert_close(child['sh_rest'], raw['sh_rest'][0], atol=0, rtol=0)

    def test_calibration_preserves_parameter_identity_and_clears_temporary_probe_gradient(self):
        raw, model, config = integration_fixture()
        MODULE.freeze_birth_radiance(model, config)
        module = NS(config=config, births=model, birth_control={}, birth_cohort=0,
                    birth_terminal=False, trainer=NS(global_step=800))
        dc, angular = model._features_dc, model._features_rest
        views = {'fit': object()}
        helper = ModuleType('pipeline.model.birth_radiance')
        def calibrate(received, cameras):
            self.assertIs(received, module); self.assertIs(cameras, views)
            with torch.enable_grad():
                dc.requires_grad_(True)
                dc.square().sum().backward()
            with torch.no_grad():
                dc.add_(.1)
            return {'before_mean_mse': 1., 'after_mean_mse': .9}
        helper.calibrate_intrinsic_dc = calibrate
        with patch.dict(sys.modules, {'pipeline.model.birth_radiance': helper}), torch.no_grad():
            MODULE.Module._calibrate_birth_radiance(module, views)
        self.assertIs(model._features_dc, dc)
        self.assertIs(model._features_rest, angular)
        self.assertFalse(dc.requires_grad or angular.requires_grad)
        self.assertIsNone(dc.grad)
        torch.testing.assert_close(angular, raw['sh_rest'], atol=0, rtol=0)
        self.assertEqual(module.birth_control['radiance_calibrations'][0]['step'], 800)

    def test_exploration_calibrates_before_metrics_and_subdivision(self):
        raw, model, config = integration_fixture()
        trace = []
        module = NS(config=config, births=model, birth_control={}, birth_cohort=0,
            birth_cohort_start=200, trainer=NS(global_step=800),
            birth_transaction=NS(fit_names=('fit',)), transaction_camera_views=lambda: {'fit': object()})
        module._calibrate_birth_radiance = lambda views: trace.append('calibrate')
        def errors(*args):
            trace.append('metrics'); return torch.tensor([1.])
        def refine(*args):
            trace.append('subdivide'); return None, {'split_parents': 0}
        with patch.object(MODULE, 'fitting_errors', errors), patch.object(MODULE, 'refine_transaction', refine):
            self.assertFalse(MODULE.Module._advance_birth_cohort(module))
        self.assertEqual(trace, ['calibrate', 'metrics', 'subdivide'])

    def test_population_calibrates_before_actual_codec_acceptance(self):
        raw, model, config = integration_fixture()
        trace = []
        trial = rb.CommitTrial(torch.tensor([0]), torch.tensor([0]), torch.tensor([0]), raw, 100, 50, 50, 0)
        module = NS(config=config, device=torch.device('cpu'), births=model, birth_control={'population_refit': {}},
            birth_cohort=2, birth_cohort_start=1400, trainer=NS(global_step=2000),
            birth_transaction=NS(fit_names=('fit',), provisional_deleted=torch.tensor([False])),
            selected_birth_state=dict(trial=trial.capture(), budget=1000, baseline_mse=[1.], selected_mse=[.8]),
            transaction_camera_views=lambda: {'fit': object()})
        module._calibrate_birth_radiance = lambda views: trace.append('calibrate')
        module._commit_population_trial = lambda chosen: trace.append('commit')
        def encode(*args):
            trace.append('encode'); return trial
        with patch.object(MODULE, 'encode_birth_trial', encode), \
                patch.object(MODULE, 'birth_validation_errors', return_value=torch.tensor([.7])), \
                patch.object(MODULE, 'fitting_errors', return_value=torch.tensor([.7])):
            MODULE.Module._finish_population_refit(module)
        self.assertEqual(trace, ['calibrate', 'encode', 'commit'])


# ----- test_family_radiance -----
class TinyFamilyGaussians(TinyGaussians):
    """Minimal mock of the canonical fixed-anchor, once-quantized contrast hook."""
    def family_quantized_raw(self, config, extent, reference, sources):
        steps = attribute_steps(config, extent)
        values = model_raw(self)
        result = {name: reference[name][sources]+quantized_symbols(
            value-reference[name][sources], steps[name])*steps[name] for name, value in values.items()}
        rows = self.family_rows
        anchor = self.family_anchor_raw['sh_dc']
        predictor = reference['sh_dc'][sources[rows[:, 0]]]
        base = predictor + quantized_symbols(anchor-predictor, steps['sh_dc']) * steps['sh_dc']
        z = quantized_symbols(self.family_contrast, steps['sh_dc']) * steps['sh_dc']
        result['sh_dc'][rows[:, 0]] = base
        result['sh_dc'][rows[:, 1]] = base+z[:, None]
        result['sh_dc'][rows[:, 2]] = base-z[:, None]
        return result


class FamilyModule(TinyModule):
    def __init__(self, novel=False):
        super().__init__()
        last = 7 if novel else 6
        raw = {name: value[3:].clone() for name, value in raw_points(last).items()}
        self.births = TinyFamilyGaussians(raw)
        self.births.family_rows = torch.tensor([[0, 1, 2]])
        self.births.family_contrast = torch.nn.Parameter(torch.zeros(1, 3), requires_grad=False)
        self.births.family_anchor_raw = {name: value[:1].clone() for name, value in raw.items()}
        self.births.source_rows = torch.full((len(raw['xyz']),), 2)
        self.births.optimizer = torch.optim.Adam([self.births._features_dc], lr=.01)
        self.birth_transaction.replacements = torch.tensor([0, 0, 0] + ([-1] if novel else []))
        self.birth_transaction.raw = model_raw(self.births)

    def make_birth_view(self):
        return QuantizedNewGaussianView(self.births, self.config.birth, 1., model_raw(self.gaussians), self.births.source_rows)


def family_fixture(novel=False):
    module = FamilyModule(novel)
    views = {}
    initial = module.births.family_contrast.detach().clone()
    for index in range(3):
        coverage = torch.tensor([[[.2, .1, .1]], [[.15, .25, .1]], [[.1, .2, .1]],
                                 [[.1, .1, .1]], [[.45, .15, .1]], [[.1, .2, .5]]])
        coverage[4] *= 1+.1*index
        if novel:
            coverage = torch.cat((coverage, torch.tensor([[[.1, .5, .2]]])))
        view = NS(camera_center=torch.zeros(3), image=torch.zeros(3, 1, 3), coverage=coverage)
        with torch.no_grad():
            module.births.family_contrast.copy_(torch.tensor([[.4, -.2, .3]]))
            view.image = module.forward(view)['render'].detach()
            module.births.family_contrast.copy_(initial)
        views[str(index)] = view
    return module, views


class FamilyMathTests(unittest.TestCase):
    def test_diagonal_sibling_mixing_looks_inconsistent_but_contrast_recovers_texture(self):
        plus = torch.tensor([[[1.]], [[.9]]], dtype=torch.float64)
        minus = torch.tensor([[[.9]], [[1.]]], dtype=torch.float64)
        true = .4
        residual = (plus-minus)*true
        point_b = (plus*residual).flatten(1).sum(-1)[:, None, None].expand(-1, 1, 3)
        point_h = plus.square().flatten(1).sum(-1)[:, None, None].expand_as(point_b)
        _, diagonal = br._coherent_dc_delta(point_b, point_h)
        k = plus-minus
        b = (k*residual).flatten(1).sum(-1)[:, None, None].expand(-1, 1, 3)
        h = k.square().flatten(1).sum(-1)[:, None, None].expand_as(b)
        delta, contrast = fr._coherent_family_delta(b, h)
        self.assertLess(float(diagonal['coherence'].max()), .01)
        self.assertGreater(float(contrast['coherence'].min()), .999)
        torch.testing.assert_close(delta, torch.full_like(delta, true), rtol=1.e-6, atol=1.e-6)

    def test_equal_or_numerically_indistinguishable_responses_cannot_amplify_detail(self):
        plus = torch.ones(1, 2, 3, dtype=torch.float64)
        active = torch.ones(1, 3, dtype=torch.bool)
        pixels = torch.ones(3, 2, 3, dtype=torch.bool)
        h = fr._contrast_response(plus, plus, active, active, pixels)
        self.assertEqual(h.tolist(), [[0., 0., 0.]])
        tiny_h = fr._contrast_response(plus, plus-1.e-8, active, active, pixels)
        delta, stats = fr._coherent_family_delta(torch.ones(2, 1, 3), tiny_h[None].expand(2, -1, -1))
        self.assertEqual(delta.tolist(), [[0., 0., 0.]])
        self.assertEqual(stats['support'].tolist(), [[0, 0, 0]])

    def test_small_but_supported_exact_contrast_has_scale_invariant_coherence(self):
        response = torch.full((3, 1, 3), 1.e-8, dtype=torch.float64)
        delta, stats = fr._coherent_family_delta(response*.4, response)
        torch.testing.assert_close(stats['coherence'], torch.ones(1, 3, dtype=torch.float64))
        torch.testing.assert_close(delta, torch.full((1, 3), .4, dtype=torch.float64))

    def test_response_contains_negative_sibling_cross_term_and_all_clamp_masks(self):
        plus = torch.tensor([[[.7, .3]]], dtype=torch.float64)
        minus = torch.tensor([[[.2, .4]]], dtype=torch.float64)
        pa = torch.tensor([[True, False, True]])
        ma = torch.tensor([[True, True, True]])
        pixels = torch.tensor([[[True, True]], [[True, True]], [[False, False]]])
        h = fr._contrast_response(plus, minus, pa, ma, pixels)
        expected = torch.tensor([[(.5**2+.1**2)*br.Y00**2, (.2**2+.4**2)*br.Y00**2, 0.]], dtype=torch.float64)
        torch.testing.assert_close(h, expected)

    def test_pair_box_and_once_quantized_contrast_preserve_opposite_increments(self):
        child_low = torch.tensor([[[-.4, -.2, -.5], [-.3, -.6, -.1]]], dtype=torch.float64)
        child_high = torch.tensor([[[.2, .3, .1], [.4, .5, .2]]], dtype=torch.float64)
        lower, upper = fr._pair_increment_bounds(child_low, child_high)
        initial = torch.tensor([[.017, -.017, .017]])
        candidate = fr._paired_box_candidate(initial, torch.tensor([[100., -100., 100.]]), 1., .05, lower, upper)
        change = (fr._decoded_contrast(candidate, .05)-fr._decoded_contrast(initial, .05)).double()
        self.assertTrue(bool(((change >= lower) & (change <= upper)).all()))
        self.assertTrue(bool((change >= child_low[:, 0]).all() & (change <= child_high[:, 0]).all()))
        self.assertTrue(bool((-change >= child_low[:, 1]).all() & (-change <= child_high[:, 1]).all()))
        torch.testing.assert_close(change+(-change), torch.zeros_like(change), rtol=0, atol=0)


class FamilyCalibrationTests(unittest.TestCase):
    def test_actual_family_model_and_quantized_hook_preserve_contrast_gradient(self):
        from pipeline.model.conservative_family import ConservativeFamilyModel
        from pipeline.model.birth_quantization import RAW_ATTRIBUTES, RAW_FIELDS

        module, views = family_fixture()
        raw = model_raw(module.births)
        model = ConservativeFamilyModel(0)
        for name, attr in zip(RAW_FIELDS, RAW_ATTRIBUTES):
            setattr(model, attr, torch.nn.Parameter(raw[name].clone(), requires_grad=False))
        anchor = {name: value[:1].clone() for name, value in raw.items()}
        anchor['logscale'] = torch.tensor([[.2, .2, .02]]).log()
        model.install_families(torch.tensor([[0, 1, 2]]), anchor,
                               tangent_offset=torch.tensor([[.07, 0.]]))
        model.source_rows = torch.tensor([2, 2, 2])
        model.optimizer = torch.optim.Adam([model.family_tangent_offset, model.family_gate_logit], lr=.01)
        module.births = model
        initial = model.family_contrast.detach().clone()
        for view in views.values():
            view.coverage[3].fill_(.8)
            with torch.no_grad():
                model.family_contrast.copy_(torch.tensor([[.4, -.2, .3]]))
                view.image = module.forward(view)['render'].detach()
                model.family_contrast.copy_(initial)
        report = fr.calibrate_family_contrast_dc(module, views)
        self.assertTrue(report['line_search']['accepted'])
        self.assertGreater(report['coherence']['min'], .999)
        torch.testing.assert_close(model.family_contrast, torch.tensor([[.4, -.2, .3]]), rtol=0, atol=1.e-5)

    def test_real_alpha_t_fit_keeps_mother_mean_parameter_flags_grad_and_optimizer(self):
        module, views = family_fixture()
        z = module.births.family_contrast
        z.grad = torch.full_like(z, 9.)
        dc = module.births._features_dc
        old_dc, old_rest = dc.clone(), module.births._features_rest.clone()
        optimizer = module.births.optimizer
        optimizer.state[dc]['exp_avg'] = torch.full_like(dc, .125)
        old_state = optimizer.state[dc]['exp_avg'].clone()
        renderer_scene, last = module.renderer.gaussians, module.renderer.last_contributions
        before = module.make_birth_view().raw_value('sh_dc').detach().clone()
        report = fr.calibrate_family_contrast_dc(module, views)
        after = module.make_birth_view().raw_value('sh_dc').detach()
        self.assertEqual(report['status'], 'accepted')
        self.assertLess(report['after_mean_mse'], report['before_mean_mse']*.001)
        self.assertIs(module.births.family_contrast, z)
        self.assertFalse(z.requires_grad)
        torch.testing.assert_close(z, torch.tensor([[.4, -.2, .3]]), rtol=0, atol=1.e-6)
        torch.testing.assert_close(z.grad, torch.full_like(z, 9.), rtol=0, atol=0)
        torch.testing.assert_close(after[0], before[0], rtol=0, atol=0)
        torch.testing.assert_close((after[1]+after[2])*.5, (before[1]+before[2])*.5, rtol=0, atol=0)
        torch.testing.assert_close(module.births._features_dc, old_dc, rtol=0, atol=0)
        torch.testing.assert_close(module.births._features_rest, old_rest, rtol=0, atol=0)
        torch.testing.assert_close(optimizer.state[dc]['exp_avg'], old_state, rtol=0, atol=0)
        self.assertIs(optimizer.param_groups[0]['params'][0], dc)
        self.assertNotIn(z, optimizer.state)
        self.assertIs(module.renderer.gaussians, renderer_scene)
        self.assertIs(module.renderer.last_contributions, last)
        json.dumps(report, allow_nan=False)

    def test_rejection_and_exception_restore_original_value_identity_and_flags(self):
        for error in (False, True):
            module, views = family_fixture()
            z, initial = module.births.family_contrast, module.births.family_contrast.clone()
            z.requires_grad_(True)
            effects = {'side_effect': RuntimeError('render failed')} if error else {'return_value': torch.ones(3)}
            with patch.object(fr, '_errors', **effects):
                if error:
                    with self.assertRaisesRegex(RuntimeError, 'render failed'):
                        fr.calibrate_family_contrast_dc(module, views)
                else:
                    report = fr.calibrate_family_contrast_dc(module, views)
                    self.assertFalse(report['line_search']['accepted'])
                    self.assertEqual(report['before_per_view_mse'], report['after_per_view_mse'])
            self.assertIs(module.births.family_contrast, z)
            self.assertTrue(z.requires_grad)
            torch.testing.assert_close(z, initial, rtol=0, atol=0)

    def test_zero_families_and_masked_out_raw_dc_skip_without_renderer(self):
        report = fr.calibrate_family_contrast_dc(NS(births=NS(family_rows=torch.empty(0, 3))), {})
        self.assertEqual(report['reason'], 'zero_families')
        report = br.calibrate_intrinsic_dc(NS(births=NS(_xyz=torch.zeros(3, 3))),
                                            {'a': object(), 'b': object()}, point_mask=torch.zeros(3, dtype=torch.bool))
        self.assertEqual(report['reason'], 'zero_selected_points')

    def test_novel_only_mask_never_changes_family_base_or_contrast(self):
        module, views = family_fixture(novel=True)
        module.births.family_contrast.data.zero_()
        initial = module.births._features_dc.detach().clone()
        for view in views.values():
            with torch.no_grad():
                module.births._features_dc[3].fill_(.4)
                view.image = module.forward(view)['render'].detach()
                module.births._features_dc.copy_(initial)
        report = br.calibrate_intrinsic_dc(module, views, point_mask=torch.tensor([False, False, False, True]))
        self.assertEqual(report['selected_candidate_count'], 1)
        self.assertTrue(report['line_search']['accepted'])
        torch.testing.assert_close(module.births._features_dc[:3], initial[:3], rtol=0, atol=0)
        torch.testing.assert_close(module.births.family_contrast, torch.zeros(1, 3), rtol=0, atol=0)
        self.assertEqual(report['support']['per_point_responding_views'][:3], [0, 0, 0])


# ----- test_conservative_family -----
def rotation(q):
    q = q/q.norm(dim=-1, keepdim=True)
    w,x,y,z = q.unbind(-1)
    return torch.stack((1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y),
        2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x),
        2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)), -1).reshape(*q.shape[:-1],3,3)


def arguments():
    return SimpleNamespace(percent_dense=.01, position_lr_init=.001,
        position_lr_final=.0001, position_lr_delay_mult=.01,
        position_lr_max_steps=100, feature_lr=.01, opacity_lr=.02,
        scaling_lr=.005, rotation_lr=.001)


def make_model(dtype=torch.float64):
    model = ConservativeFamilyModel(1)
    generator = torch.Generator().manual_seed(41)
    raw = dict(xyz=torch.randn(8,3,generator=generator,dtype=dtype),
        rotation=torch.tensor([[1.,0.,0.,0.]],dtype=dtype).repeat(8,1),
        logscale=torch.full((8,3),-3.,dtype=dtype),
        opacity=torch.full((8,1),-.3,dtype=dtype),
        sh_dc=torch.randn(8,1,3,generator=generator,dtype=dtype),
        sh_rest=torch.randn(8,3,3,generator=generator,dtype=dtype))
    for name,attr in zip(RAW_FIELDS,RAW_ATTRIBUTES):setattr(model,attr,nn.Parameter(raw[name].clone()))
    model.active_sh_degree=1;model.spatial_lr_scale=2.
    model.source_rows=torch.arange(8);model.max_radii2D=torch.zeros(8,dtype=dtype)
    anchor={name:value[:2].detach().clone() for name,value in raw.items()}
    anchor['xyz']=torch.tensor([[.3,.4,2.],[-.2,.1,1.]],dtype=dtype)
    anchor['logscale']=torch.tensor([[.2,.08,.002],[.004,.09,.18]],dtype=dtype).log()
    anchor['rotation']=torch.tensor([[.7,.2,-.3,.1],[.5,.1,.3,-.2]],dtype=dtype)
    anchor['rotation']/=anchor['rotation'].norm(dim=-1,keepdim=True)
    anchor['opacity']=torch.logit(torch.tensor([[.8],[.35]],dtype=dtype))
    model.install_families(torch.tensor([[0,3,6],[1,4,7]]),anchor,
        tangent_offset=torch.tensor([[.06,.025],[.03,-.02]],dtype=dtype),
        gate_logit=torch.tensor([[-.3],[.7]],dtype=dtype),
        contrast_dc=torch.tensor([[.03,-.02,.04],[-.01,.05,-.02]],dtype=dtype))
    return model


def family_statistics(model):
    raw=model.family_phenotype();rows=model.family_rows
    xyz=raw['xyz'][rows]
    scales=raw['logscale'][rows].exp()
    q=raw['rotation'][rows];R=rotation(q)
    cov=R@torch.diag_embed(scales.square())@R.transpose(-1,-2)
    axes=model._family_tangent_axes[:,None].expand(-1,3,-1)
    area=scales.gather(2,axes).prod(-1)
    mass=area*optical_depth_integral(raw['opacity'][rows].sigmoid().squeeze(-1))
    weights=mass/mass.sum(-1,keepdim=True)
    mean=(xyz*weights[...,None]).sum(1)
    delta=xyz-mean[:,None]
    mixture=((cov+delta[...,None]*delta[...,None,:])*weights[...,None,None]).sum(1)
    return mean,mixture,mass


class OpticalMassTests(unittest.TestCase):
    def test_known_values_cap_extension_inverse_and_monotonicity(self):
        a=torch.tensor([0.,.2,.5,.99,1.],dtype=torch.float64)
        value=optical_depth_integral(a)
        self.assertAlmostEqual(float(value[0]),0.,places=14)
        self.assertAlmostEqual(float(value[2]),math.pi**2/12-math.log(2)**2/2,places=12)
        self.assertAlmostEqual(float(value[-1]),OPTICAL_MASS_MAX,places=13)
        torch.testing.assert_close(inverse_optical_depth_integral(value),a,rtol=0,atol=2e-11)
        dense=torch.linspace(0,1,10001,dtype=torch.float64)
        self.assertTrue(bool((optical_depth_integral(dense).diff()>0).all()))
        self.assertTrue(bool((inverse_optical_depth_integral(dense*OPTICAL_MASS_MAX).diff()>0).all()))

    def test_analytic_backward_matches_finite_differences(self):
        alpha=torch.tensor([.001,.2,.6,.95,.995],dtype=torch.float64,requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(optical_depth_integral,(alpha,),eps=1e-6,atol=2e-7,rtol=2e-5))
        mass=torch.tensor([.001,.1,.7,1.5,1.62],dtype=torch.float64,requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(inverse_optical_depth_integral,(mass,),eps=1e-6,atol=2e-7,rtol=2e-5))


class ConservativeFamilyTests(unittest.TestCase):
    def test_continuous_optical_mass_moments_and_parent_geometry(self):
        model=make_model();mean,cov,mass=family_statistics(model);anchor=model.family_anchor_raw
        target=model._family_parent_area*optical_depth_integral(anchor['opacity'].sigmoid().flatten())
        torch.testing.assert_close(mass.sum(-1),target,rtol=2e-10,atol=2e-13)
        torch.testing.assert_close(mass[:,1],mass[:,2],rtol=0,atol=0)
        torch.testing.assert_close(mean,anchor['xyz'],rtol=0,atol=1e-12)
        R=rotation(anchor['rotation']);expected=R@torch.diag_embed(anchor['logscale'].exp().square())@R.transpose(-1,-2)
        torch.testing.assert_close(cov,expected,rtol=1e-10,atol=1e-12)
        raw=model.family_phenotype()
        for name in ('xyz','rotation','logscale'):
            torch.testing.assert_close(raw[name][model.family_rows[:,0]],anchor[name],rtol=0,atol=0)

    def test_normal_does_not_drift_and_offset_is_inside_one_sigma(self):
        model=make_model();raw=model.family_phenotype();rows=model.family_rows
        normal=model._family_frame.gather(2,model._family_normal_axis[:,None,None].expand(-1,3,1)).squeeze(-1)
        delta=raw['xyz'][rows[:,1:]]-model.family_anchor_raw['xyz'][:,None]
        torch.testing.assert_close((delta*normal[:,None]).sum(-1),torch.zeros(2,2,dtype=torch.float64),rtol=0,atol=2e-16)
        local=torch.einsum('fji,fkj->fki',model._family_frame,delta)
        radius=(local/model.family_anchor_raw['logscale'].exp()[:,None]).square().sum(-1)
        self.assertTrue(bool((radius<1).all()))
        child_R=rotation(raw['rotation'][rows[:,1]])
        child_normal=child_R.gather(2,model._family_normal_axis[:,None,None].expand(-1,3,1)).squeeze(-1)
        torch.testing.assert_close(child_normal,normal,rtol=0,atol=3e-16)

    def test_signed_radiance_and_shared_sh(self):
        model=make_model();raw=model.family_phenotype();dc=raw['sh_dc'][model.family_rows]
        torch.testing.assert_close((dc[:,1]+dc[:,2])*.5,dc[:,0],rtol=0,atol=1e-16)
        torch.testing.assert_close(dc[:,1]-dc[:,0],model.family_contrast[:,None],rtol=0,atol=1e-16)
        rest=raw['sh_rest'][model.family_rows]
        torch.testing.assert_close(rest,model.family_anchor_raw['sh_rest'][:,None].expand_as(rest),rtol=0,atol=0)

    def test_latent_finite_differences_and_no_independent_family_raw_gradient(self):
        model=make_model()
        def phenotype(offset,gate,contrast):
            model.family_gate_logit=gate
            values=model._family_values(offset,contrast)
            return torch.cat([values[name].flatten() for name in RAW_FIELDS])
        inputs=tuple(x.detach().clone().requires_grad_(True) for x in
                     (model.family_tangent_offset,model.family_gate_logit,model.family_contrast))
        self.assertTrue(torch.autograd.gradcheck(phenotype,inputs,eps=1e-6,atol=3e-6,rtol=3e-4))
        model=make_model();model.training_setup(arguments())
        loss=sum((value.square().sum() for value in model.family_phenotype().values()))
        loss.backward()
        self.assertGreater(float(model.family_tangent_offset.grad.abs().sum()),0)
        self.assertGreater(float(model.family_gate_logit.grad.abs().sum()),0)
        self.assertIsNone(model.family_contrast.grad)
        for attr in RAW_ATTRIBUTES:
            grad=getattr(model,attr).grad
            torch.testing.assert_close(grad[model.family_rows.flatten()],torch.zeros_like(grad[model.family_rows.flatten()]),rtol=0,atol=0)
            self.assertGreater(float(grad[[2,5]].abs().sum()),0)
        names={g['name'] for g in model.optimizer.param_groups}
        self.assertIn('family_tangent_offset',names);self.assertIn('family_gate_logit',names)
        self.assertNotIn('family_contrast',names)

    def test_zero_gate_reverts_parent_and_saturated_offset_returns_mass(self):
        model=make_model()
        with torch.no_grad():model.family_gate_logit.fill_(-float('inf'))
        raw=model.family_phenotype();alpha=raw['opacity'][model.family_rows].sigmoid()
        torch.testing.assert_close(alpha[:,0],model.family_anchor_raw['opacity'].sigmoid(),rtol=1e-10,atol=1e-12)
        self.assertTrue(bool((alpha[:,1:]<1e-40).all()))
        self.assertTrue(bool(torch.isfinite(raw['opacity']).all()))
        with torch.no_grad():
            model.family_gate_logit.fill_(float('inf'));model.family_tangent_offset.fill_(1000.)
        _,_,mass=family_statistics(model)
        target=model._family_parent_area*optical_depth_integral(model.family_anchor_raw['opacity'].sigmoid().flatten())
        torch.testing.assert_close(mass.sum(-1),target,rtol=1e-9,atol=1e-12)
        self.assertTrue(bool((mass[:,0]>0).all()))
        torch.testing.assert_close(model.get_opacity[model.family_rows[:,1]],torch.ones(2,1,dtype=torch.float64),rtol=0,atol=1e-12)

    def test_isotropic_zero_offset_has_finite_gradients(self):
        model=make_model()
        anchor={k:v[:1].clone() for k,v in model.family_anchor_raw.items()}
        anchor['logscale']=torch.tensor([[.1,.1,.001]],dtype=torch.float64).log()
        model.install_families([[0,3,6]],anchor)
        sum(x.square().sum() for x in model.family_phenotype().values()).backward()
        self.assertTrue(bool(torch.isfinite(model.family_tangent_offset.grad).all()))
        self.assertTrue(bool(torch.isfinite(model.family_gate_logit.grad).all()))

    def test_quantization_pairs_contrast_and_derives_pose_on_shared_plane(self):
        model=make_model();model.family_contrast.requires_grad_(True)
        config=SimpleNamespace(new_xyz_step_extent=.013,new_sh_step=.01,
            new_logscale_step=.001,new_opacity_step=.01)
        reference={name:value[:3].detach().clone() for name,value in model.family_phenotype().items()}
        sources=torch.tensor([0,1,2,0,1,2,0,1])
        raw=model.family_quantized_raw(config,2.,reference,sources)
        dc=raw['sh_dc'][model.family_rows]
        torch.testing.assert_close((dc[:,1]+dc[:,2])*.5,dc[:,0],rtol=0,atol=1e-16)
        torch.testing.assert_close((dc[:,1]-dc[:,0])/.01,torch.round(model.family_contrast/.01)[:,None],rtol=0,atol=1e-14)
        dc[:,1].sum().backward()
        torch.testing.assert_close(model.family_contrast.grad,torch.ones_like(model.family_contrast),rtol=0,atol=0)
        normal=model._family_frame.gather(2,model._family_normal_axis[:,None,None].expand(-1,3,1)).squeeze(-1)
        delta=raw['xyz'][model.family_rows[:,1:]]-model.family_anchor_raw['xyz'][:,None]
        torch.testing.assert_close((delta*normal[:,None]).sum(-1),torch.zeros(2,2,dtype=torch.float64),rtol=0,atol=2e-16)
        with self.assertRaisesRegex(ValueError,'shared surviving source'):
            inconsistent=sources.clone();inconsistent[3]=1
            model.family_quantized_raw(config,2.,reference,inconsistent)

    def test_capture_restore_has_no_alias_and_preserves_latent_adam(self):
        model=make_model();model.training_setup(arguments())
        sum(x.square().sum() for x in model.family_phenotype().values()).backward();model.optimizer.step();model.optimizer.zero_grad()
        expected={k:v.detach().clone() for k,v in model.family_phenotype().items()};packet=model.capture()
        restored=ConservativeFamilyModel(1).restore(packet,arguments(),device='cpu')
        for name in RAW_FIELDS:torch.testing.assert_close(restored.birth_parameter_value(name),expected[name],rtol=0,atol=0)
        for old,new in zip(model.optimizer.param_groups,restored.optimizer.param_groups):
            self.assertEqual(old['name'],new['name'])
            for k,v in model.optimizer.state[old['params'][0]].items():
                torch.testing.assert_close(restored.optimizer.state[new['params'][0]][k],v,rtol=0,atol=0)
        with torch.no_grad():
            packet['family']['anchor_raw']['xyz'].add_(5);packet['family']['tangent_offset'].add_(7);packet['base'][1].add_(4)
        for name in RAW_FIELDS:torch.testing.assert_close(restored.birth_parameter_value(name),expected[name],rtol=0,atol=0)

    def test_atomic_subset_remaps_family_and_retains_independent_state(self):
        model=make_model();model.training_setup(arguments())
        sum(x.square().sum() for x in model.family_phenotype().values()).backward();model.optimizer.step()
        ids=torch.tensor([2,6,3,0]);subset=model.family_subset(ids)
        torch.testing.assert_close(subset.family_rows,torch.tensor([[3,2,1]]),rtol=0,atol=0)
        for name in RAW_FIELDS:torch.testing.assert_close(subset.birth_parameter_value(name),model.birth_parameter_value(name)[ids],rtol=0,atol=0)
        torch.testing.assert_close(subset.source_rows,model.source_rows[ids],rtol=0,atol=0)
        with self.assertRaisesRegex(ValueError,'atomically'):model.family_subset([0,3])
        before=subset.family_tangent_offset.detach().clone()
        with torch.no_grad():model.family_tangent_offset.add_(1)
        torch.testing.assert_close(subset.family_tangent_offset,before,rtol=0,atol=0)

    def test_plain_capture_restore_and_factory_install(self):
        model=make_model();base=model.capture()['base']
        restored=ConservativeFamilyModel(1).restore(base,arguments())
        self.assertEqual(len(restored.family_rows),0)
        state=model.family_state();install_conservative_families(restored,state)
        for name in RAW_FIELDS:torch.testing.assert_close(restored.birth_parameter_value(name),model.birth_parameter_value(name),rtol=0,atol=0)

    def test_gate_phase_freeze_and_restore_keeps_optimizer_group(self):
        model=make_model();model.family_gate_logit.requires_grad_(False)
        model.training_setup(arguments())
        loss=sum(x.square().sum() for x in model.family_phenotype().values())
        loss.backward()
        self.assertIsNone(model.family_gate_logit.grad)
        self.assertGreater(float(model.family_tangent_offset.grad.abs().sum()),0)
        restored=ConservativeFamilyModel(1).restore(model.capture(),arguments())
        self.assertFalse(restored.family_gate_logit.requires_grad)
        self.assertIn('family_gate_logit',{g['name'] for g in restored.optimizer.param_groups})
        restored.family_gate_logit.requires_grad_(True)
        sum(x.square().sum() for x in restored.family_phenotype().values()).backward()
        self.assertTrue(bool(torch.isfinite(restored.family_gate_logit.grad).all()))
        self.assertGreater(float(restored.family_gate_logit.grad.abs().sum()),0)

    def test_float32_mass_and_thin_normal_remain_numerically_stable(self):
        model=make_model(torch.float32)
        mean,cov,mass=family_statistics(model)
        target=model._family_parent_area*optical_depth_integral(model.family_anchor_raw['opacity'].sigmoid().flatten())
        torch.testing.assert_close(mass.sum(-1),target,rtol=3e-6,atol=1e-9)
        torch.testing.assert_close(mean,model.family_anchor_raw['xyz'],rtol=0,atol=3e-7)
        self.assertTrue(bool(torch.isfinite(cov).all()))

    def test_precomputed_covariance_uses_derived_family_rotation(self):
        model=make_model();raw=model.family_phenotype()
        R=rotation(raw['rotation']);matrix=R@torch.diag_embed(raw['logscale'].exp().square())@R.transpose(-1,-2)
        packed=torch.stack((matrix[:,0,0],matrix[:,0,1],matrix[:,0,2],
                            matrix[:,1,1],matrix[:,1,2],matrix[:,2,2]),-1)
        torch.testing.assert_close(model.get_covariance(),packed,rtol=0,atol=0)

    def test_refinement_seed_preserves_world_tangent_direction_for_axis_permutations(self):
        from pipeline.model import residual_birth as rb
        from pipeline.model.gaussian import BirthGaussianModel
        # Every normal-axis position and both orders of unequal tangent scales
        # must carry the split's world direction into the family unchanged.
        for scales in itertools.permutations((.002,.08,.2)):
            with self.subTest(scales=scales):
                anchor={k:v[:1].clone() for k,v in make_model(torch.float32).family_anchor_raw.items()}
                anchor['logscale']=torch.tensor([scales]).log()
                parent=BirthGaussianModel(1)
                for name,attr in zip(RAW_FIELDS,RAW_ATTRIBUTES):
                    setattr(parent,attr,nn.Parameter(anchor[name].clone()))
                normal=int(anchor['logscale'][0].argmin())
                tangent=[axis for axis in range(3) if axis != normal]
                local=torch.zeros(3);local[tangent]=torch.tensor([.015,-.025])
                world_direction=rotation(anchor['rotation'])[0]@local
                children=[{name:value[0].clone() for name,value in anchor.items()} for _ in range(2)]
                children[0]['xyz']+=world_direction;children[1]['xyz']-=world_direction
                transaction=SimpleNamespace(support_domains=[{'kind':'covariance'}],
                    replacements=torch.tensor([5]),sources=torch.tensor([0]),
                    scores=torch.ones(1),proposed_spatial=0,proposed_repairs=1,
                    fit_names=('fit.png',),validation_names=(),group_ids=torch.tensor([0]),
                    region_diagnostics={})
                module=SimpleNamespace(births=parent,birth_transaction=transaction,
                    config=SimpleNamespace(sh_degree=1),device=torch.device('cpu'),
                    make_birth_view=lambda:SimpleNamespace(raw_value=lambda name:anchor[name]))
                with patch.object(rb,'refinement_evidence',return_value=[{}]), \
                        patch.object(rb,'split_support',return_value=(children,[{},{}],1.)):
                    refined,_=rb.refine_conservative_transaction(module,SimpleNamespace(max_candidates=3),{})
                model=ConservativeFamilyModel(1)
                for name,attr in zip(RAW_FIELDS,RAW_ATTRIBUTES):
                    setattr(model,attr,nn.Parameter(refined.raw[name].clone()))
                model.install_family_state(refined.family_state)
                expected_offset=local[tangent]
                torch.testing.assert_close(model.family_tangent_offset[0],expected_offset,rtol=0,atol=1e-7)
                bounded=world_direction/(1+(expected_offset/torch.tensor(scales)[tangent]).square().sum()).sqrt()
                actual=model.get_xyz[1]-anchor['xyz'][0]
                torch.testing.assert_close(actual,bounded,rtol=0,atol=1e-7)
                self.assertGreater(float(torch.dot(actual,world_direction)/(actual.norm()*world_direction.norm())),.999999)


# ----- test_family_controller -----
def family_make_model(xyz):
    model = BirthGaussianModel(1)
    n = len(xyz)
    raw = dict(xyz=xyz.clone(), rotation=torch.tensor([[1., 0., 0., 0.]]).repeat(n, 1),
        logscale=torch.tensor([[.2, .15, .002]]).log().repeat(n, 1), opacity=torch.full((n, 1), .3),
        sh_dc=torch.full((n, 1, 3), .1234567), sh_rest=torch.full((n, 3, 3), .013579))
    for name, attr in zip(RAW_FIELDS, RAW_ATTRIBUTES):
        setattr(model, attr, torch.nn.Parameter(raw[name].clone()))
    model.active_sh_degree=1; model.spatial_lr_scale=1.
    model.max_radii2D=torch.zeros(n)
    return model


class FamilyControllerTests(unittest.TestCase):
    def test_typed_novel_and_previous_train_access_never_touch_future_or_test(self):
        residual = torch.ones(2, 2)*.2
        current = torch.ones(3, 2, 2)*.5
        typed, report = rb.typed_novel_residual(residual, current, current, torch.zeros(2, 2))
        self.assertEqual(float(typed.max()), 0)
        typed, _ = rb.typed_novel_residual(residual, current, current, torch.ones(2, 2))
        torch.testing.assert_close(typed, residual)
        seen=[]
        data=NS(current_frame=2)
        class Dataset:
            items=[NS(path=Path('cam01.png'))]
            def __getitem__(self, index):
                seen.append(data.current_frame)
                return NS(to=lambda device: NS(image=torch.ones(3,2,2)))
        data.get_train_dataset=lambda: Dataset()
        module=NS(datamodule=data, device=torch.device('cpu'))
        rb.previous_training_views(module, ['cam01.png'])
        self.assertEqual(seen, [1]); self.assertEqual(data.current_frame, 2)
        with self.assertRaises(ValueError): rb.previous_training_views(module, ['missing.png'])
        self.assertEqual(data.current_frame, 2)

    def test_spatial_contrast_ranking_rejects_uniform_dc_bias(self):
        base=model_raw(family_make_model(torch.tensor([[-.4, 0., 3.], [.4, 0., 3.]])))
        h=w=96
        x=torch.arange(w)[None].expand(h,-1)
        r=torch.full((3,h,w), .1)
        r[:, :, 48:] += (x[:, 48:]-58)[None]*.005
        views=[NS(path=f'fit{i}.png',image=torch.zeros(3,h,w),world_view_transform=torch.eye(4),
                  fovx=2*math.atan(w/160),fovy=2*math.atan(h/160)) for i in range(3)]
        cache={str(view.path):(r, torch.ones(2,dtype=torch.bool)) for view in views}
        order, report=rb.rank_repair_contrast(NS(),views,torch.arange(2),base,cache)
        self.assertEqual(order.tolist(), [1])
        self.assertEqual(report['positive_contrast_count'], 1)

    def test_family_refine_phase_sources_codec_and_terminal_snapshot_are_consistent(self):
        old_xyz=torch.cartesian_prod(torch.tensor([.1,.3,.6,.9]),torch.tensor([.1,.9]),torch.tensor([.1,.9])).repeat_interleave(2,0)
        old=family_make_model(old_xyz); coarse=family_make_model(torch.tensor([[.3,.4,.7]]))
        coarse.source_rows=torch.tensor([1])
        raw=model_raw(coarse)
        transaction=rb.BirthTransaction(raw,torch.tensor([1]),torch.tensor([1]),torch.ones(1),
            raw['xyz'],raw['rotation'],raw['logscale'].exp(),torch.tensor([3]),
            ('fit1','fit2'),('fit1','fit2'),0,1,torch.tensor([17]),{},[rb.covariance_support(raw,0)])
        config=MODULE.ModuleConfig(birth=MODULE.BirthConfig(mechanism='residual_transaction'))
        motion=SpatialGatedMotion(RigidTreeConfig(motion_steps=200,birth_steps=600,rate_loss_weight=0.,relative_motion_loss_weight=0.))
        motion.setup(old._xyz.detach(),old._rotation.detach(),1.,frame_index=1)
        module=NS(device=torch.device('cpu'),current_frame=1,config=config,gaussians=old,births=coarse,
            birth_transaction=transaction,birth_terminal=False,datamodule=NS(cameras_extent=1.),
            motion=motion,opacity_update=None)
        module.make_birth_view=lambda:MODULE.Module.make_birth_view(module)
        evidence=[dict(valid=torch.ones(9,dtype=torch.bool),error=torch.ones(9),
            color=torch.tensor([[.9,.1,.1]]*4+[[.1,.1,.9]]*5))]
        with patch.object(rb,'refinement_evidence',return_value=evidence):
            refined,report=rb.refine_transaction(module,config.birth.residual,{})
        self.assertEqual(len(refined.sources),3)
        self.assertEqual(refined.group_ids.tolist(),[17,17,17])
        self.assertEqual(refined.replacements.tolist(),[1,1,1])
        MODULE.Module._install_birth_transaction(module,refined)
        self.assertFalse(module.births.family_gate_logit.requires_grad)
        self.assertFalse(module.births.family_contrast.requires_grad)
        qview=module.make_birth_view()
        loss=(qview.get_xyz[1]-torch.tensor([.4,.5,.7])).square().sum()
        loss.backward(); module.births.optimizer.step(); module.births.optimizer.zero_grad(set_to_none=True)
        self.assertIn(module.births.family_tangent_offset,module.births.optimizer.state)
        self.assertNotIn(module.births.family_gate_logit,module.births.optimizer.state)
        with torch.no_grad():module.births.family_contrast.fill_(.12345)
        delete=torch.zeros(len(old_xyz),dtype=torch.bool);delete[1]=True
        trial=rb.encode_trial(module,torch.ones(3,dtype=torch.bool),delete)
        self.assertEqual(len(trial.sources.unique()),1)
        selected=rb.selected_population(module,trial)
        self.assertEqual(selected.family_state['rows'].tolist(),[[0,1,2]])
        snapshot=selected.capture()
        module.birth_terminal=True
        MODULE.Module._install_birth_transaction(module,rb.BirthTransaction.restore(snapshot,'cpu'))
        self.assertTrue(module.births.family_gate_logit.requires_grad)
        self.assertEqual(len(module.births.optimizer.state),0)
        qview=module.make_birth_view()
        syms=qview.symbols['sh_dc']
        torch.testing.assert_close((syms[1]-syms[0])+(syms[2]-syms[0]),torch.zeros_like(syms[0]),atol=0,rtol=0)
        materialized,_=materialize_quantized_gaussians(selected.fixed_reference,qview)
        for name in RAW_FIELDS:
            torch.testing.assert_close(materialized[name],trial.raw[name],atol=0,rtol=0)
        diagnostics=family_quantization_diagnostics(module.births,qview)
        self.assertTrue(diagnostics['shared_source'])
        self.assertEqual(diagnostics['contrast_symbol_cancellation_max'],0)
        # Whole-root selection is mandatory; a two-member wire population is invalid.
        with self.assertRaises(ValueError):module.births.family_subset(torch.tensor([0,1]))


# ----- test_predictive_colour -----
class PredictiveSimplexTests(unittest.TestCase):
    def test_overlapping_whole_image_basis_recovers_joint_not_diagonal_fit(self):
        prior = torch.full((3, 2, 2), .2, dtype=torch.float64)
        basis = torch.tensor([[.10, .08, .02, .01], [.09, .07, .03, -.01],
                              [.06, -.02, .10, .08]], dtype=torch.float64).repeat(1, 3)
        expected = torch.tensor([.2, .3, .1], dtype=torch.float64)
        ends = [prior+(b.reshape_as(prior)) for b in basis]
        target = prior+(expected@basis).reshape_as(prior)
        moments = PredictiveColourMoments();moments.add(prior, *ends, target)
        got, report = moments.solve()
        torch.testing.assert_close(got, expected, rtol=0, atol=1e-10)
        g, b = torch.tensor(report['gram']), torch.tensor(report['rhs'])
        self.assertGreater(float(g[0, 1]), 0)
        self.assertGreater(float((b/g.diag()-expected.float()).abs().max()), .1)
        self.assertLess(abs(report['fitted_surrogate_mse']), 1e-15)

    def test_sum_boundary_active_face_is_analytic(self):
        w, _ = solve_predictive_simplex(torch.eye(3), torch.tensor([1., .75, -.2]))
        torch.testing.assert_close(w, torch.tensor([.625, .375, 0.], dtype=torch.float64), rtol=0, atol=1e-12)

    def test_prior_and_rank_deficient_endpoints_do_not_amplify(self):
        w, report = solve_predictive_simplex(torch.zeros(3, 3), torch.zeros(3))
        self.assertEqual(w.tolist(), [0., 0., 0.]);self.assertEqual(report['gram_rank'], 0)
        g = torch.ones(3, 3, dtype=torch.float64)
        w, _ = solve_predictive_simplex(g, torch.full((3,), .6))
        self.assertAlmostEqual(float(w.sum()), .6, places=7)
        self.assertTrue(bool((w >= 0).all()))
        w, _ = solve_predictive_simplex(torch.eye(3), torch.tensor([-1., -2., -3.]))
        self.assertEqual(w.tolist(), [0., 0., 0.])

    def test_equal_view_weighting_is_not_pixel_count_weighting(self):
        m = PredictiveColourMoments()
        for n, delta in ((1, .2), (20, .4)):
            p = torch.full((3, 1, n), .1)
            m.add(p, p+.5, p, p, p+delta)
        w, r = m.solve()
        self.assertAlmostEqual(float(w[0]), .6, places=6)
        self.assertEqual(r['view_count'], 2)

    def test_full_rank_and_singular_solutions_satisfy_convex_optimality(self):
        generator = torch.Generator().manual_seed(719)
        for rank in (1, 2, 3):
            for _ in range(10):
                a = torch.randn(rank, 3, dtype=torch.float64, generator=generator)
                y = torch.randn(rank, dtype=torch.float64, generator=generator)
                g, b = a.T@a, a.T@y
                w, _ = solve_predictive_simplex(g, b)
                self.assertTrue(bool((w >= 0).all()));self.assertLessEqual(float(w.sum()), 1+1e-12)
                gradient = g@w-b
                active = w > 1e-9
                # A KKT certificate is an independent global-optimum check
                # for this convex problem, including rank-deficient Grams.
                multiplier = -gradient[active].mean() if bool(active.any()) and float(w.sum()) > 1-1e-9 else 0.
                self.assertGreaterEqual(float(multiplier), -1e-9)
                self.assertTrue(bool((gradient+multiplier >= -1e-9).all()))
                self.assertLess(float(((gradient+multiplier)*w).abs().max()), 1e-9)

    def test_invalid_gram_and_images_fail_before_mutation(self):
        with self.assertRaises(ValueError):solve_predictive_simplex(torch.diag(torch.tensor([-1., 1., 1.])), torch.zeros(3))
        m = PredictiveColourMoments();p = torch.zeros(3, 1, 1)
        with self.assertRaises(ValueError):m.add(p, p, p, p, p+float('nan'))


class PredictiveCalibrationTests(unittest.TestCase):
    def fixture(self):
        view = NS(image=torch.full((3, 1, 1), .6))
        ends = dict(prior=torch.full_like(view.image, .2), dc=torch.full_like(view.image, .6),
                    sh=torch.full_like(view.image, .2), full=torch.full_like(view.image, .2))
        return view, lambda view, kind: {'render': ends[kind]}

    def test_native_nonlinearity_backtracks_and_applies_only_final_weight(self):
        view, render = self.fixture();applied=[];calls=[]
        def validate(w):
            calls.append(w.clone())
            # The endpoint surrogate wants D=1, but raw blend/codec has a
            # different nonlinear response. Native D=.5 improves; D=1 hurts.
            return dict(mse=.04 if float(w[0]) > .75 else (.01 if float(w[0]) > 0 else .02), codec='same_predictor')
        report = calibrate_predictive_colour({'cam01.png':view}, render, applied.append, validate)
        self.assertEqual(len(applied), 1)
        self.assertAlmostEqual(float(applied[0][0]), .5, places=6)
        self.assertTrue(report['nonprior_selected'])
        self.assertEqual(report['selected_validation']['codec'], 'same_predictor')
        self.assertEqual(len(calls), 3)

    def test_codec_rejection_returns_exact_prior_without_training(self):
        view, render = self.fixture();applied=[]
        def validate(w):return dict(mse=.01, eligible=not bool(w.any()))
        report = calibrate_predictive_colour([view], render, applied.append, validate)
        self.assertEqual(applied[0].tolist(), [0., 0., 0.])
        self.assertFalse(report['nonprior_selected'])
        self.assertEqual(report['prior_weight'], 1.)

    def test_external_test_camera_and_missing_actual_validation_are_rejected(self):
        view, render = self.fixture()
        with self.assertRaises(ValueError):calibrate_predictive_colour({'cam00.png':view}, render, lambda w:None, lambda w:0)
        with self.assertRaises(ValueError):calibrate_predictive_colour([view], render, lambda w:None, None)

    def test_source_residual_labels_preserve_three_direction_weights_and_report(self):
        view, _ = self.fixture();kinds=('prior','common','detail','full')
        values={'prior':.2,'common':.2,'detail':.6,'full':.2}
        requested=[];applied=[]
        def render(view, kind):
            requested.append(kind)
            return torch.full_like(view.image, values[kind])
        def validate(w):return (.2+.4*float(w[1])-.6)**2
        report=calibrate_predictive_colour([view],render,applied.append,validate,endpoint_kinds=kinds)
        self.assertEqual(requested,list(kinds))
        self.assertAlmostEqual(float(applied[0][1]),1.,places=6)
        self.assertEqual(report['endpoint_order'],list(kinds))
        self.assertEqual(report['weight_labels'],list(kinds[1:]))
        self.assertAlmostEqual(report['weights_by_endpoint']['detail'],1.,places=6)
        self.assertAlmostEqual(sum(report['weights_by_endpoint'].values()),1.,places=12)
        self.assertNotIn('dc',report['weights_by_endpoint'])

    def test_default_endpoint_names_and_invalid_names_do_not_change_legacy_contract(self):
        view, render=self.fixture();applied=[]
        report=calibrate_predictive_colour([view],render,applied.append,lambda w:(1-float(w[0]))**2)
        self.assertEqual(report['endpoint_order'],['prior','dc','sh','full'])
        self.assertEqual(report['weight_labels'],['dc','sh','full'])
        for invalid in (('prior','dc','sh'),('prior','common_only','common_only','full'),
                        ('wrong_prior','dc','sh','full'),('prior','dc',None,'full')):
            with self.assertRaisesRegex(ValueError,'endpoint_kinds'):
                calibrate_predictive_colour([view],render,applied.append,lambda w:0,endpoint_kinds=invalid)


if __name__ == '__main__':
    unittest.main()
