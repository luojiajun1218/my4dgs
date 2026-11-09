"""CPU contracts for regional birth topology and admission."""
import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from types import SimpleNamespace as NS
from unittest.mock import patch

import numpy as np
import torch

from pipeline.frame_plan import fitting_old_motion
from pipeline.model.birth_quantization import RAW_FIELDS, model_raw
from pipeline.model.frame_plan_birth import FramePlanBirthModel
from pipeline.model.gaussian import BirthGaussianModel
from pipeline.model.residual_birth import RawView
from pipeline.model.spatial_gate import spatial_layout, spatial_state_route
from pipeline.module import BirthConfig, Module
from pipeline.regional_birth_admission import (_prospective_decoded_scene, checked_training_pool,
    leaf_groups, reserve_admission_cameras, verify_encoded_admission)
from pipeline.regional_topology import (assign_measurement_units, native_colour_evidence,
    select_region_candidates, route_seed_units, save_region_diagnostics)
from pipeline.trainer import Trainer
from scripts.replay_rigid_tree import decode_frame
from tests import test_frame_plan as fixed_tests
from tests.test_frame_plan import controller_fixture, cpu_gaussian_setup


def paths_for(*leaves):
    # D5 breadth-first octree IDs; `leaves` are base-8 leaf codes.
    leaves = np.asarray(leaves, dtype=np.int64)
    return np.stack([sum(8 ** i for i in range(level)) +
                     (leaves >> (3 * (5 - level))) for level in range(1, 6)], 1)


class RegionalTopologyTests(unittest.TestCase):
    def test_real_compositing_l1_and_unassigned_conserve_pixel_error(self):
        module, views, _, weights = support_fixture(constant=True)
        cache = [dict(view=v, residual=v.rendered-v.image,
                      native_mass=weights.sum((1, 2))) for v in views]
        original = module.renderer.gaussians
        module.renderer.collect_contributions = True
        numerator, denominator, uncovered = native_colour_evidence(module, cache)
        expected = weights.sum((1, 2)).numpy()
        np.testing.assert_allclose(denominator, np.tile(expected, (3, 1)), rtol=1e-6)
        np.testing.assert_allclose(numerator, np.tile(expected * .1, (3, 1)), rtol=1e-6)
        self.assertAlmostEqual(uncovered[0]['unassigned_l1'], 64 * .4 * .1, places=5)
        self.assertAlmostEqual(uncovered[0]['attributed_l1'] + uncovered[0]['unassigned_l1'],
                               uncovered[0]['total_l1'], places=5)
        self.assertIs(module.renderer.gaussians, original)
        self.assertTrue(module.renderer.collect_contributions)
        views[0].path = 'cam00.png'
        with self.assertRaisesRegex(ValueError, 'external test'):
            native_colour_evidence(module, cache)

    def test_equal_pixel_error_same_score_at_different_region_sizes(self):
        paths = paths_for(0, 8)
        denom = np.array([[64., 320.], [64., 320.]])
        units = assign_measurement_units(paths, .04 * denom, denom)
        self.assertEqual(list(units['unit_depth']), [5, 5])
        np.testing.assert_allclose(units['unit_score'], [.04, .04])
        np.testing.assert_array_equal(units['unit_reason'], [2, 2])

    def test_weak_siblings_merge_raw_sums_with_no_duplicate(self):
        # Leaf 0 resolves alone. Weak 1 and 2 merge at D4 using only their
        # pooled numerators/denominators, not the resolved sibling's data.
        paths = paths_for(0, 1, 2)
        den = np.array([[100., 30., 10.], [100., 30., 10.]])
        num = den * np.array([[.02, .06, .02], [.02, .06, .02]])
        units = assign_measurement_units(paths, num, den)
        np.testing.assert_allclose(units['unit_numerator'].sum(0), num.sum(1))
        np.testing.assert_allclose(units['unit_denominator'].sum(0), den.sum(1))
        d5 = np.flatnonzero(units['unit_depth'] == 5)
        d4 = np.flatnonzero(units['unit_depth'] == 4)
        self.assertEqual((len(d5), len(d4)), (1, 1))
        self.assertAlmostEqual(units['unit_score'][d5[0]], .02)
        self.assertAlmostEqual(units['unit_score'][d4[0]], .05)
        self.assertEqual(units['leaf_unit'][1], units['leaf_unit'][2])
        self.assertNotEqual(units['leaf_unit'][0], units['leaf_unit'][1])
        self.assertEqual(units['unit_reason'][d4[0]], 2)

    def test_insufficient_and_unobserved_are_never_low_error(self):
        paths = paths_for(0, 1, 2)
        den = np.array([[1., 0., 0.], [1., 0., 0.]])
        units = assign_measurement_units(paths, den * .9, den)
        self.assertEqual(len(units['unit_reason']), 1)
        self.assertEqual(units['unit_reason'][0], 0)
        self.assertTrue(np.isnan(units['unit_score'][0]))
        np.testing.assert_array_equal(select_region_candidates(units, [10., 10., 10.], [2, 2, 2], 9), [])

    def test_views_are_normalized_separately_and_require_two(self):
        paths = paths_for(0)
        den = np.array([[32.], [320.], [2.]])
        num = den * np.array([[.01], [.07], [1.]])
        units = assign_measurement_units(paths, num, den)
        self.assertAlmostEqual(units['unit_score'][0], .04)
        self.assertEqual(units['unit_reason'][0], 2)
        fewer = assign_measurement_units(paths, num, np.array([[32.], [2.], [2.]]))
        self.assertEqual(fewer['unit_reason'][0], 0)

    def test_triggered_residual_mass_and_support_with_no_quota_fill(self):
        paths = paths_for(0, 0, 0, 8, 8, 16)
        den = np.full((2, 6), 64.)
        num = den * np.array([[.05, .05, .05, .04, .04, .01]] * 2)
        units = assign_measurement_units(paths, num, den)
        rows = select_region_candidates(units, [100, 90, 80, 5, 4, 1000], [2, 2, 2, 2, 1, 2], 3)
        self.assertEqual(rows.tolist(), [0, 1, 2])
        # Changing only region IDs does not grant extra selection budget.
        original = units['point_unit'].copy()
        units['point_unit'][:5] = original[0]
        self.assertEqual(select_region_candidates(
            units, [100, 90, 80, 5, 4, 1000], [2, 2, 2, 2, 1, 2], 3).tolist(), [0, 1, 2])
        np.testing.assert_array_equal(select_region_candidates(units, [0]*6, [2]*6, 8), [])

    def test_routing_and_sidecar_preserve_leaf_to_partition(self):
        xyz = torch.tensor([[-1., -1., -1.], [1., 1., 1.], [1.1, 1.1, 1.1]])
        paths, parents, opened, centers, routing = spatial_layout(xyz, 5, return_centers=True,
                                                                    return_routing=True)
        layout = SimpleNamespace(spatial_paths=paths, spatial_centers=centers,
                                 **{k:routing.get(k) for k in ('spatial_support_ball',
                                    'spatial_leaf_centers', 'spatial_leaf_nodes')})
        den = np.full((2, 3), 64.)
        units = assign_measurement_units(paths.numpy(), den*.04, den)
        leaf, group = route_seed_units(layout, xyz, units)
        np.testing.assert_array_equal(leaf, paths[:, -1])
        np.testing.assert_array_equal(group, units['point_unit'])
        with tempfile.TemporaryDirectory() as directory:
            module = SimpleNamespace(output_dir=Path(directory), current_frame=3)
            path = save_region_diagnostics(module, 200, units,
                dict(camera_names=['cam01.png','cam02.png'], threshold=.03,
                     min_support=32, min_views=2), seed_rows=np.array([1]))
            with np.load(Path(directory)/path, allow_pickle=False) as data:
                np.testing.assert_array_equal(data['leaf_unit'], units['leaf_unit'])
                np.testing.assert_array_equal(data['unit_numerator'], units['unit_numerator'])
                np.testing.assert_array_equal(data['seed_rows'], [1])

    def test_birth_path_restricts_regions_alpha_and_quantized_position(self):
        with tempfile.TemporaryDirectory() as directory:
            module, views = controller_fixture(directory)
            module.config.birth.regional_color_birth = True
            module.config.birth.residual_support_init = True
            module.config.birth.regional_color_min_support = 32.
            module.config.birth.regional_color_min_views = 2
            module.config.birth.regional_color_l1_threshold = .03
            module.config.birth.regional_color_depth = 5
            module.config.birth.regional_color_reference_layout = None
            module.births = FramePlanBirthModel(1, logit_opacity=True)
            module.motion_committed = True
            module.early_initialized = True
            module._plan_used_quota = 0
            module._plan_events = [{}]
            module.birth_control['early_replacement'] = {}
            module._plan_step_views = list(views.values())[:4]
            xyz = module.gaussians._xyz.detach()
            paths, parents, opened, centers, routing = spatial_layout(xyz, 5, return_centers=True,
                                                                       return_routing=True)
            layout = SimpleNamespace(spatial_paths=paths, spatial_centers=centers,
                **{k:routing.get(k) for k in ('spatial_support_ball','spatial_leaf_centers','spatial_leaf_nodes')})
            module._regional_color_layout = (layout, 'synthetic fixed frame-zero layout')

            def evidence(proposal, cache):
                positions = proposal.gaussians.get_xyz.detach().cpu().numpy()
                den = np.full((len(cache), len(positions)), 40.)
                num = den * np.where(positions[:, 0] < .5, .1, .01)[None]
                return num, den, [dict(camera=x['name']) for x in cache]

            def geometry(raw, rows, cache, step):
                seed = {key:raw[key][rows].clone() for key in ('xyz','rotation','logscale')}
                if len(rows):
                    seed['xyz'][0] = raw['xyz'][-1]
                return seed, torch.ones(len(rows), dtype=torch.bool), dict(policy='synthetic')

            with patch('pipeline.regional_topology.native_colour_evidence', side_effect=evidence), \
                 patch('pipeline.frame_plan.residual_support_seed', side_effect=geometry):
                module._plan_birth(200)
            event = module._plan_events[-1]['regional_color']
            self.assertGreater(event['candidate_count'], 0)
            self.assertEqual(event['boundary_rejected_count'], 1)
            self.assertGreater(event['boundary_rejected_l1'], 0)
            self.assertGreater(event['born'], 0)
            sources = module.births.parent_rows
            self.assertTrue(bool((xyz[sources, 0] < .5).all()))
            actual = module.births.cohorts[0]['parameters']['_opacity'].sigmoid()
            torch.testing.assert_close(actual, torch.full_like(actual, .1), rtol=0, atol=1e-7)
            with np.load(Path(directory) / event['diagnostic_sidecar'], allow_pickle=False) as data:
                np.testing.assert_array_equal(data['seed_rows'], sources.numpy())
                self.assertTrue(np.all(data['unit_reason'][data['seed_unit']] == 2))

    def test_post200_alpha_image_gradient(self):
        with tempfile.TemporaryDirectory() as directory:
            module, views = controller_fixture(directory)
            module.config.birth.keep_incumbent_fixed_after_motion = True
            module.config.birth.keep_incumbent_alpha_after_motion = True
            module.config.rigid_tree.spatial_gate.final_joint_refine_steps = 0
            module.motion_committed = True
            module.early_initialized = True
            module._plan_prepare_rates(201)
            self.assertTrue(module.opacity_update.delta.requires_grad)
            self.assertFalse(fitting_old_motion(module))
            module.gaussians.rigid_alpha = module.opacity_update.alphas()
            image = module.renderer(next(iter(views.values())), module.background)['render']
            (image - .55).square().mean().backward()
            self.assertIsNotNone(module.opacity_update.delta.grad)
            self.assertGreater(float(module.opacity_update.delta.grad.norm()), 0.)


class AdmissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_single_training_view_keeps_proposal_and_validation_disjoint(self):
        with tempfile.TemporaryDirectory() as directory:
            module, _ = controller_fixture(directory)
            module.config.birth.regional_postfit_admission = True
            training = module.frame_plan_training_views(200, 1)
            proposal = {Path(view.path).name for view in module._plan_step_views}
            self.assertEqual(len(training), 1)
            self.assertEqual(len(proposal), 4)
            reserve_admission_cameras(module)
            reserved = set(module._plan_admission_camera_names)
            self.assertFalse(proposal & reserved)
            for step in (201, 250, 300):
                with self.subTest(step=step):
                    training = module.frame_plan_training_views(step, 1)
                    selected = {Path(view.path).name for view in module._plan_step_views}
                    self.assertEqual(len(training), 1)
                    self.assertFalse(selected & reserved)
                    self.assertEqual(len(selected), 4 if step == 250 else 1)

    def test_config_contract(self):
        self.assertFalse(BirthConfig().regional_postfit_admission)
        with self.assertRaisesRegex(ValueError, 'must be boolean'):
            BirthConfig(regional_postfit_admission=1).validate(200)
        with self.assertRaisesRegex(ValueError, 'fixed incumbent'):
            BirthConfig(frame_training_plan=True, regional_postfit_admission=True).validate(200)
        BirthConfig(frame_training_plan=True, keep_incumbent_fixed_after_motion=True,
                    regional_postfit_admission=True).validate(200)

    def test_reservation_is_disjoint_and_does_not_touch_rng(self):
        with tempfile.TemporaryDirectory() as directory:
            m, views = controller_fixture(directory)
            m.config.birth.regional_postfit_admission = True
            state = torch.get_rng_state().clone()
            first = m.frame_plan_training_views(200, 4)
            proposal = {Path(v.path).name for v in first}
            reserve_admission_cameras(m)
            reserved = set(m._plan_admission_camera_names)
            self.assertEqual(len(reserved), 4)
            self.assertFalse(proposal & reserved)
            self.assertNotIn('cam00.png', reserved)
            self.assertTrue(torch.equal(state, torch.get_rng_state()))
            expected = []
            for step in range(201, 301):
                names = [Path(v.path).name for v in m.frame_plan_training_views(step, 4)]
                self.assertFalse(set(names) & reserved)
                expected.append(names)
            # Reservation is stateless in the frame/proposal and survives replay.
            m.frame_plan_training_views(200, 4)
            reserve_admission_cameras(m)
            self.assertEqual(reserved, set(m._plan_admission_camera_names))
            self.assertEqual(expected, [[Path(v.path).name for v in m.frame_plan_training_views(step, 4)]
                                        for step in range(201, 301)])
            with self.assertRaisesRegex(ValueError, 'at least eight'):
                checked_training_pool(m, dict(list(views.items())[:7]))
            m.datamodule.config.training_camera_names = list(views)[:8]
            with self.assertRaisesRegex(ValueError, 'configured training split'):
                checked_training_pool(m, views)

    def test_all_leaves_and_chunks_covered_once_in_row_order(self):
        # Two fixed leaves, with interleaved input; first leaf needs 3 chunks.
        leaves = torch.tensor([10, 11] * 70 + [10, 10, 10])
        paths = torch.stack((torch.ones(len(leaves), dtype=torch.long), leaves), 1)
        opened = torch.zeros(12, dtype=torch.bool)
        opened[1] = True
        groups = leaf_groups(paths, opened)
        self.assertEqual([leaf for leaf, _, _ in groups], [10, 10, 10, 11, 11, 11])
        self.assertEqual([len(rows) for _, _, rows in groups], [32, 32, 9, 32, 32, 6])
        self.assertTrue(all(owner == leaf for leaf, owner, _ in groups))
        all_rows = torch.cat([rows for _, _, rows in groups]).sort().values
        self.assertTrue(torch.equal(all_rows, torch.arange(len(leaves))))
        for leaf, _, rows in groups:
            self.assertTrue(bool((leaves[rows] == leaf).all()))
            self.assertTrue(bool((rows[1:] > rows[:-1]).all()))
        opened[1] = False
        closed = leaf_groups(paths, opened)
        self.assertEqual([leaf for leaf, _, _ in closed], [10, 10, 10, 11, 11, 11])
        self.assertTrue(all(owner == 1 for _, owner, _ in closed))

    def test_wire_verifier_rejects_signed_zero_drift(self):
        expected = {name: torch.zeros(1, dtype=torch.float32) for name in RAW_FIELDS}
        actual = {name: value.clone() for name, value in expected.items()}
        actual['xyz'][0] = -0.
        self.assertTrue(torch.equal(expected['xyz'], actual['xyz']))
        decoded = NS(digest=lambda: b'codec-state', birth_sources=torch.empty(0, dtype=torch.long))
        module = NS(_plan_admission_verification=(expected, torch.empty(0, dtype=torch.long), [], []),
                    _plan_diagnostic_reference={}, output_dir=Path('/tmp'), current_frame=1,
                    motion=NS(frame_reference_state=object(), codec_previous_state=decoded),
                    trainer=NS(), device='cpu', gaussians=object())
        with patch('pipeline.regional_birth_admission.model_raw', return_value=expected), \
                patch('scripts.replay_rigid_tree.decode_frame', return_value=(decoded, actual)):
            with self.assertRaisesRegex(ValueError, 'xyz attributes'):
                verify_encoded_admission(module)

    def test_zero_effect_and_subtolerance_noise_are_rejected(self):
        from pipeline.regional_birth_admission import admit_fitted_births, _tolerance
        views = {f'cam{i:02}.png': NS(path=f'cam{i:02}.png') for i in range(1, 9)}
        births = NS(_xyz=torch.zeros(3, 3), source_rows=torch.zeros(3, dtype=torch.long))
        def select(rows, optimizer):
            births.source_rows = births.source_rows[rows]
            births._xyz = births._xyz[rows]
        births.select_rows_inplace = select
        module = NS(transaction_camera_views=lambda: views, device='cpu',
                    datamodule=NS(config=NS(training_camera_names=None, validation_camera_names=None)),
                    _plan_admission_camera_names=list(views)[:4], trainer=NS(),
                    gaussians=NS(_xyz=torch.zeros(1, 3)), births=births,
                    motion=NS(optimizer=None), birth_control={})
        raw = {name: torch.zeros(4, 3) for name in RAW_FIELDS}
        decoded = NS(xyz=torch.zeros(4, 3), spatial_open=torch.ones(4, dtype=torch.bool))
        groups = [(i, i, torch.tensor([i])) for i in range(3)]
        baseline = .01
        tol = _tolerance(baseline, baseline)
        measured = [[baseline]*4, [baseline]*4, [baseline-tol/4]*4,
                    [baseline+2*tol, baseline-6*tol, baseline-6*tol, baseline-6*tol]]
        with patch('pipeline.regional_birth_admission._prospective_decoded_scene',
                   return_value=(raw, decoded, torch.zeros(3, dtype=torch.long))), \
                patch('pipeline.regional_birth_admission.spatial_state_route'), \
                patch('pipeline.regional_birth_admission.leaf_groups', return_value=groups), \
                patch('pipeline.regional_birth_admission._mse_per_view', side_effect=measured):
            admit_fitted_births(module)
        report = module.birth_control['regional_postfit_admission']
        self.assertEqual([g['accepted'] for g in report['groups']], [False, False, True])
        self.assertEqual(report['accepted_count'], 1)
        self.assertEqual(report['rejected_count'], 2)
        self.assertEqual(report['groups'][0]['improvement'], 0.)
        self.assertLess(report['groups'][1]['improvement'], report['groups'][1]['numerical_tolerance'])
        self.assertGreater(report['groups'][2]['trial_per_view'][0],
                           report['groups'][2]['baseline_per_view'][0])

    def test_300_step_checkpoint_groups_and_true_codec_replay(self):
        # Sign-coded CPU compositor gives the second cohort a positive effect
        # and the first a negative one on every reserved validation camera.
        class SignRenderer:
            def __init__(self, old_count, gaussians):
                self.old_count, self.gaussians = old_count, gaussians
                self.collect_contributions = False

            def __call__(self, view, background):
                model = self.gaussians
                alpha = model.get_opacity[self.old_count:, 0]
                dc = model.get_features[self.old_count:, 0, 0]
                value = .5 + .05 * (alpha * dc).sum()
                return {'render': value.expand_as(view.image)}

        case = fixed_tests.FixedIncumbentTests()
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as resume_dir, \
                patch.object(BirthGaussianModel, 'training_setup', cpu_gaussian_setup):
            m, trainer, prior, initial = case.setup_controller(directory, True)
            m.config.birth.regional_postfit_admission = True
            trainer._views_per_step = lambda step: 1
            for step in range(1, 201):
                case.step(m, trainer, step)
            names = list(m._plan_admission_camera_names)
            self.assertFalse(set(names) & set(m._plan_events[0]['proposal_cameras']))
            checkpoint = dict(global_step=200, current_frame=1)
            Module.on_save_checkpoint(m, checkpoint)
            resumed, resumed_trainer, _, _ = case.setup_controller(resume_dir, True)
            resumed.config.birth.regional_postfit_admission = True
            resumed_trainer._views_per_step = lambda step: 1
            Module.on_load_checkpoint(resumed, copy.deepcopy(checkpoint))
            self.assertEqual(resumed._plan_admission_camera_names, names)
            for step in range(201, 300):
                case.step(m, trainer, step)
                self.assertFalse(set(names) & {Path(v.path).name for v in m._plan_step_views})
                if step in (201, 250):
                    resumed.frame_plan_training_views(step, 1)
                    self.assertEqual([v.path for v in m._plan_step_views],
                                     [v.path for v in resumed._plan_step_views])
            self.assertFalse(set(names) & set(m._plan_events[1]['proposal_cameras']))
            old_count = len(m.gaussians._xyz)
            self.assertGreaterEqual(len(m.births.cohorts), 2)
            self.assertTrue(all(len(c['parameters']['_xyz']) for c in m.births.cohorts))
            state = m.motion.frame_reference_state
            paths = spatial_state_route(state.xyz, state)
            leaves = paths[:, -1]
            distinct = torch.unique(leaves, sorted=True)
            self.assertGreaterEqual(len(distinct), 2)
            positions = [state.xyz[leaves.eq(leaf).nonzero(as_tuple=True)[0][0]]
                         for leaf in (distinct[0], distinct[-1])]
            with torch.no_grad():
                for index, batch in enumerate(m.births.cohorts):
                    batch['parameters']['_xyz'][:] = positions[index > 0]
                    batch['parameters']['_features_dc'].fill_(-1. if index == 0 else 1.)
                    batch['parameters']['_opacity'].fill_(.2 if not m.births.logit_opacity else -1.38629436)
                first_count = len(m.births.cohorts[0]['parameters']['_xyz'])
                candidate_count = len(m.births._xyz)
                # A second-cohort child's proposal parent is the first newborn,
                # but its actual codec predictor still addresses an old row.
                m.births.parent_rows[first_count:] = old_count

            original_cleanup = m._plan_cleanup
            def prepare_validation(step):
                original_cleanup(step)
                if step != 300:
                    return
                renderer = SignRenderer(old_count, m.gaussians)
                m.renderer = renderer
                before_digest = m.motion.frame_reference_state.digest()
                previous_state = m.motion.codec_previous_state.digest()
                decoded_raw, decoded, sources = _prospective_decoded_scene(m)
                self.assertEqual(before_digest, m.motion.frame_reference_state.digest())
                self.assertEqual(previous_state, m.motion.codec_previous_state.digest())
                self.assertFalse((Path(directory) / 'frame_storage' / 'rigid_tree' / 'motion_001.bin').exists())
                self.assertTrue(bool((sources < old_count).all()))
                self.assertTrue(bool((m.births.parent_rows[first_count:] >= old_count).all()))
                target = {name: torch.cat((value[:old_count], value[old_count + first_count:]))
                          for name, value in decoded_raw.items()}
                renderer.gaussians = RawView(target, m.gaussians.active_sh_degree, m.gaussians)
                for name in names:
                    view = m.transaction_camera_views()[name]
                    view.image = renderer(view, m.background)['render'].detach().clone()
                renderer.gaussians = m.gaussians
            m._plan_cleanup = prepare_validation
            case.step(m, trainer, 300)
            self.assertFalse(set(names) & {Path(v.path).name for v in m._plan_step_views})
            report = m.birth_control['regional_postfit_admission']
            self.assertEqual(report['candidates'], candidate_count)
            self.assertEqual(sum(g['candidate_count'] for g in report['groups']), candidate_count)
            self.assertTrue(any(g['accepted'] for g in report['groups']))
            self.assertTrue(any(not g['accepted'] for g in report['groups']))
            self.assertEqual(report['accepted_count'], candidate_count-first_count)
            self.assertTrue(all(report['codec_verification']['six_attributes_exact'].values()))
            self.assertTrue(report['codec_verification']['source_rows_exact'])
            self.assertTrue(report['codec_verification']['independent_stream_decoded'])
            current = report['old_only_mean_mse']
            for group in report['groups']:
                self.assertEqual(current, group['baseline_mse'])
                if group['accepted']:
                    self.assertGreater(group['improvement'], group['numerical_tolerance'])
                    current = group['trial_mse']
            self.assertEqual(current, report['accepted_mean_mse'])
            self.assertEqual(m.frame_report['birth_count'], candidate_count-first_count)
            output = Path(directory) / 'frame_storage' / 'rigid_tree'
            decoded, raw = decode_frame(output, 1, prior, initial)
            self.assertEqual(decoded.digest(), m.motion.codec_previous_state.digest())
            actual = model_raw(m.gaussians)
            for name in RAW_FIELDS:
                self.assertTrue(torch.equal(raw[name], actual[name].cpu()), name)

    def test_midframe_checkpoint_restores_camera_split_and_final_packets(self):
        case = fixed_tests.FixedIncumbentTests()
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as resume_dir, \
                patch.object(BirthGaussianModel, 'training_setup', cpu_gaussian_setup):
            m, trainer, _, _ = case.setup_controller(directory, True)
            m.config.birth.regional_postfit_admission = True
            trainer._frame_diagnostic_seconds = 0.
            Trainer.snapshot_frame_plan_reference(trainer)
            initial_reference = {name: value.clone() for name, value in m._plan_diagnostic_reference.items()}
            for step in range(1, 251):
                case.step(m, trainer, step)
            packet = dict(global_step=250, current_frame=1)
            Module.on_save_checkpoint(m, packet)
            resumed, resumed_trainer, _, _ = case.setup_controller(resume_dir, True)
            resumed.config.birth.regional_postfit_admission = True
            Module.on_load_checkpoint(resumed, copy.deepcopy(packet))
            self.assertEqual(resumed._plan_admission_camera_names, m._plan_admission_camera_names)
            for name in RAW_FIELDS:
                self.assertTrue(torch.equal(resumed._plan_diagnostic_reference[name], initial_reference[name]))
            for step in range(251, 301):
                case.step(m, trainer, step)
                case.step(resumed, resumed_trainer, step)
                self.assertEqual([v.path for v in m._plan_step_views],
                                 [v.path for v in resumed._plan_step_views])
            self.assertEqual(m.birth_control['regional_postfit_admission']['accepted_count'],
                             resumed.birth_control['regional_postfit_admission']['accepted_count'])
            for name in RAW_FIELDS:
                self.assertTrue(torch.equal(model_raw(m.gaussians)[name], model_raw(resumed.gaussians)[name]), name)
            first = Path(directory) / 'frame_storage' / 'rigid_tree'
            second = Path(resume_dir) / 'frame_storage' / 'rigid_tree'
            packets = lambda folder: {p.name: p.read_bytes() for p in folder.glob('*_001.bin')}
            self.assertEqual(packets(first), packets(second))

    def test_empty_births_after_cleanup_still_encode_and_verify(self):
        case = fixed_tests.FixedIncumbentTests()
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(BirthGaussianModel, 'training_setup', cpu_gaussian_setup):
            m, trainer, _, _ = case.setup_controller(directory, True)
            m.config.birth.regional_postfit_admission = True
            for step in range(1, 300):
                case.step(m, trainer, step)
            cleanup = m._plan_cleanup
            def empty_at_cleanup(step):
                if step == 300:
                    for batch in m.births.cohorts:
                        batch['parameters']['_opacity'].data.fill_(-100. if m.births.logit_opacity else 0.)
                cleanup(step)
            m._plan_cleanup = empty_at_cleanup
            case.step(m, trainer, 300)
            report = m.birth_control['regional_postfit_admission']
            self.assertEqual((report['candidates'], report['group_count'], report['accepted_count']), (0, 0, 0))
            self.assertEqual(m.frame_report['birth_count'], 0)
            self.assertTrue(all(report['codec_verification']['six_attributes_exact'].values()))
            self.assertTrue(report['codec_verification']['source_rows_exact'])


# ----- Regional alpha repair -----

import math
from types import MethodType, SimpleNamespace


from pipeline.model.birth_quantization import RAW_FIELDS, materialize_quantized_gaussians, model_raw
from tests.test_frame_plan import checkpoint_harness, controller_fixture, cpu_gaussian_setup


class RegionalAlphaRepairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def fixture(self, directory, *, retire=False):
        module, views = controller_fixture(directory)
        module.config.birth.regional_color_birth = True
        module.config.birth.regional_alpha_repair = True
        module.config.birth.keep_incumbent_fixed_after_motion = True
        module.config.birth.keep_incumbent_alpha_after_motion = True
        module.config.rigid_tree.spatial_gate.final_joint_refine_steps = 0
        module.config.opacity_update.l1_weight = .05
        module.config.opacity_update.proximal_regularization = True
        module.motion_committed = True
        module.early_initialized = True
        module.births = FramePlanBirthModel(1, logit_opacity=True)
        module._plan_events = []
        module._plan_regional_alpha_state = None
        checkpoint_harness(module)
        parameter = module.opacity_update.delta
        with torch.no_grad():
            parameter.fill_(.04)
        module.motion.optimizer.state[parameter] = dict(step=torch.tensor(200.),
            exp_avg=torch.full_like(parameter, .2), exp_avg_sq=torch.full_like(parameter, .3))
        module.motion.prepare_plan_step(200)
        if retire:
            with torch.no_grad():
                parameter[0] = -module.opacity_update.alpha_reference[0]
            module._plan_cleanup(200)
        return module, views

    @staticmethod
    def units(module, triggered, *, new_points=0):
        count = len(module.gaussians._xyz)
        point_unit = np.zeros(count + new_points, dtype=np.int64)
        point_unit[np.asarray(triggered, dtype=np.int64)] = 1
        if new_points:
            point_unit[count:] = 2  # triggered new rows must never enable old rows
        return dict(point_unit=point_unit, unit_reason=np.array([1, 2, 2], dtype=np.int8))

    def image_step(self, module, views, step):
        module.trainer.global_step = step
        module._plan_prepare_rates(step)
        module.motion.optimizer.zero_grad(set_to_none=True)
        module.gaussians.rigid_alpha = module.opacity_update.alphas()
        image = module.renderer(next(iter(views.values())), module.background)['render']
        (image - .55).square().mean().backward()
        module.gaussians.rigid_alpha = None
        gradient = module.opacity_update.delta.grad.detach().clone()
        module._plan_regional_alpha_mask_gradient(step)
        return gradient

    def adam_and_prox(self, module, step):
        update = module.opacity_update
        module.motion.optimizer.step()
        update.optimizer_step(.00001, proximal=True, perform_step=False,
                              external_state=module.motion.optimizer.state[update.delta])
        module._plan_regional_alpha_restore(step)
        module.motion.optimizer.zero_grad(set_to_none=True)

    def test_mask_maps_survivors_and_freezes_parameter_moments_alpha_symbols(self):
        with tempfile.TemporaryDirectory() as directory:
            module, views = self.fixture(directory, retire=True)
            module._plan_regional_alpha_select(200, self.units(module, [0, 2], new_points=3))
            update, state = module.opacity_update, module._plan_regional_alpha_state
            rows = update.rows.clone()
            eligible = state['eligible']
            self.assertEqual(eligible.nonzero().flatten().tolist(), [int(rows[0]), int(rows[2])])
            self.assertFalse(bool(eligible[0]))  # old source row 0 was retired
            baseline_delta = update.delta.detach().clone()
            baseline_alpha = update.alphas().detach().clone()
            baseline_symbols = update.symbols().clone()
            adam = module.motion.optimizer.state[update.delta]
            initial_moments = {k: adam[k].clone() for k in ('exp_avg', 'exp_avg_sq')}
            gradient = self.image_step(module, views, 201)
            self.assertGreater(float(gradient[eligible].abs().sum()), 0.)
            self.assertGreater(float(gradient[~eligible].abs().sum()), 0.)
            self.assertEqual(float(update.delta.grad[~eligible].abs().sum()), 0.)
            self.adam_and_prox(module, 201)
            self.assertTrue(torch.equal(update.delta[~eligible], baseline_delta[~eligible]))
            self.assertTrue(torch.equal(update.alphas().detach()[~eligible[rows]], baseline_alpha[~eligible[rows]]))
            self.assertTrue(torch.equal(update.symbols()[~eligible], baseline_symbols[~eligible]))
            for key in initial_moments:
                self.assertTrue(torch.equal(adam[key][~eligible], initial_moments[key][~eligible]))
            self.assertGreater(float((update.delta[eligible] - baseline_delta[eligible]).abs().sum()), 0.)
            self.assertEqual(int(adam['step']), 201)  # scalar Adam clock remains shared
            audit = module.birth_control['regional_alpha_repair']['invariant_checks'][-1]
            for key in ('max_frozen_alpha_change', 'max_frozen_symbol_change',
                        'max_frozen_delta_change', 'max_frozen_moment_change'):
                self.assertEqual(audit[key], 0.)

    def test_250_unions_eligibility_and_reactivates_from_step200_state(self):
        with tempfile.TemporaryDirectory() as directory:
            module, views = self.fixture(directory, retire=True)
            module._plan_regional_alpha_select(200, self.units(module, [0]))
            state = module._plan_regional_alpha_state
            old_rows = module.opacity_update.rows
            new_row = int(old_rows[1])
            original_moment = state['baseline_moments']['exp_avg'][new_row].clone()
            self.image_step(module, views, 201)
            self.adam_and_prox(module, 201)
            torch.testing.assert_close(module.opacity_update.delta[new_row], state['baseline_delta'][new_row], rtol=0, atol=0)
            torch.testing.assert_close(module.motion.optimizer.state[module.opacity_update.delta]['exp_avg'][new_row],
                                       original_moment, rtol=0, atol=0)
            module._plan_regional_alpha_select(250, self.units(module, [1], new_points=2))
            self.assertTrue(bool(state['eligible'][old_rows[0]]))
            self.assertTrue(bool(state['eligible'][new_row]))
            event = module.birth_control['regional_alpha_repair']['mask_events'][-1]
            self.assertEqual(event['newly_eligible_old_count'], 1)
            self.assertEqual(event['eligible_old_count'], 2)
            gradient = self.image_step(module, views, 251)
            self.assertGreater(float(gradient[new_row].abs().sum()), 0.)
            self.adam_and_prox(module, 251)
            self.assertGreater(float((module.opacity_update.delta[new_row]-state['baseline_delta'][new_row]).abs()), 0.)

    def test_no_trigger_freezes_all_and_full_update_has_no_mask(self):
        with tempfile.TemporaryDirectory() as directory:
            module, views = self.fixture(directory)
            module._plan_regional_alpha_select(200, self.units(module, []))
            baseline = module.opacity_update.delta.detach().clone()
            self.image_step(module, views, 201)
            self.adam_and_prox(module, 201)
            self.assertTrue(torch.equal(module.opacity_update.delta, baseline))
            self.assertEqual(module.birth_control['regional_alpha_repair']['mask_events'][0]['eligible_old_count'], 0)
        with tempfile.TemporaryDirectory() as directory:
            module, _ = self.fixture(directory)
            module.full_update_active = True
            module._plan_regional_alpha_select(200, self.units(module, []))
            self.assertIsNone(module._plan_regional_alpha_state)
            parameter = module.opacity_update.delta
            parameter.grad = torch.ones_like(parameter)
            module._plan_regional_alpha_mask_gradient(201)
            self.assertTrue(bool(parameter.grad.eq(1).all()))
            before = parameter.detach().clone()
            module.motion.optimizer.step()
            module._plan_regional_alpha_restore(201)
            self.assertGreater(float((parameter-before).abs().sum()), 0.)

    def test_config_requires_regional_frame_plan_and_postmotion_alpha(self):
        self.assertFalse(BirthConfig().regional_alpha_repair)
        for values in (dict(), dict(frame_training_plan=True),
                       dict(frame_training_plan=True, regional_color_birth=True)):
            with self.subTest(values=values), self.assertRaisesRegex(ValueError, 'Regional alpha repair'):
                BirthConfig(regional_alpha_repair=True, **values).validate(200)
        BirthConfig(frame_training_plan=True, regional_color_birth=True,
                    keep_incumbent_alpha_after_motion=True, residual_support_init=True,
                    regional_alpha_repair=True).validate(200)

    def test_selection_uses_real_birth_units_even_with_zero_quota(self):
        with tempfile.TemporaryDirectory() as directory:
            module, views = self.fixture(directory)
            module.config.birth.residual_support_init = True
            module.config.birth.max_new_gaussians = 0
            module.config.birth.residual.max_candidates = 0
            module.config.birth.regional_color_min_support = 32.
            module.config.birth.regional_color_min_views = 2
            module.config.birth.regional_color_l1_threshold = .03
            module._plan_used_quota = 0
            module._plan_events = [{}]
            module.birth_control['early_replacement'] = {}
            module._plan_step_views = list(views.values())[:4]
            xyz = module.gaussians._xyz.detach()
            paths, parents, opened, centers, routing = spatial_layout(xyz, 5, return_centers=True,
                                                                       return_routing=True)
            layout = SimpleNamespace(spatial_paths=paths, spatial_centers=centers,
                **{k:routing.get(k) for k in ('spatial_support_ball','spatial_leaf_centers','spatial_leaf_nodes')})
            module._regional_color_layout = (layout, 'fixed fixture layout')
            current_step = [200]
            def evidence(proposal, cache):
                positions = proposal.gaussians.get_xyz.detach().cpu().numpy()
                den = np.full((len(cache), len(positions)), 40.)
                trigger = positions[:, 0] < .5 if current_step[0] == 200 else positions[:, 0] >= .5
                return den*np.where(trigger, .1, .01)[None], den, [dict(camera=x['name']) for x in cache]
            with patch('pipeline.regional_topology.native_colour_evidence', side_effect=evidence):
                module._plan_birth(200)
                state = module._plan_regional_alpha_state
                expected = xyz[:, 0] < .5
                self.assertTrue(torch.equal(state['eligible'][module.opacity_update.rows], expected))
                self.assertEqual(module._plan_events[-1]['born'], 0)
                # The 250 statistics use the complete old+new scene, but the
                # alpha mask can only refer to current old predecessor rows.
                reference = module._adaptive_reference()
                seed = {name:value[[0]].clone() for name,value in reference.items()}
                seed['opacity'].fill_(torch.logit(torch.tensor(.1)))
                module.births.add(seed, torch.tensor([0]), torch.tensor([0]), 200,
                                  module.motion.optimizer, module.config.gaussian_stage2)
                module._plan_events.append({})
                current_step[0] = 250
                module._plan_birth(250)
                self.assertTrue(bool(state['eligible'][module.opacity_update.rows].all()))
                self.assertEqual(len(state['eligible']), len(module.opacity_update.delta))
                self.assertEqual(module._plan_events[-1]['born'], 0)
                events = module.birth_control['regional_alpha_repair']['mask_events']
                self.assertEqual([row['step'] for row in events], [200, 250])
                self.assertEqual(events[-1]['old_count'], len(xyz))
                self.assertEqual(events[-1]['eligible_old_count'], len(xyz))

    def test_step250_checkpoint_restore_next_update_is_identical(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(
                BirthGaussianModel, 'training_setup', cpu_gaussian_setup):
            module, views = self.fixture(directory, retire=True)
            module._plan_regional_alpha_select(200, self.units(module, [0, 2]))
            self.image_step(module, views, 201)
            self.adam_and_prox(module, 201)
            module._plan_regional_alpha_select(250, self.units(module, [1], new_points=2))
            module.early_shared_steps = module.trainer.global_step = 250
            packet = dict(global_step=250, current_frame=1)
            Module.on_save_checkpoint(module, packet)
            clone, clone_views = self.fixture(directory)
            Module.on_load_checkpoint(clone, copy.deepcopy(packet))
            for instance, camera in ((module, views), (clone, clone_views)):
                instance.early_shared_steps = 250
                instance.trainer.global_step = 251
                instance._plan_prepare_rates(251)
                instance.motion.optimizer.zero_grad(set_to_none=True)
                instance.gaussians.rigid_alpha = instance.opacity_update.alphas()
                image = instance.renderer(next(iter(camera.values())), instance.background)['render']
                (image - .55).square().mean().backward()
                instance.gaussians.rigid_alpha = None
                instance._plan_post_step([], 251)
            self.assertTrue(torch.equal(module.opacity_update.delta, clone.opacity_update.delta))
            self.assertTrue(torch.equal(module.opacity_update.alphas(), clone.opacity_update.alphas()))
            self.assertTrue(torch.equal(module._plan_regional_alpha_state['eligible'],
                                        clone._plan_regional_alpha_state['eligible']))
            for key in ('exp_avg', 'exp_avg_sq', 'step'):
                a = module.motion.optimizer.state[module.opacity_update.delta][key]
                b = clone.motion.optimizer.state[clone.opacity_update.delta][key]
                self.assertTrue(torch.equal(a, b), key)

    def test_full_frame_retirement_resume_and_real_six_attribute_decode(self):
        def configure(module):
            module.config.birth.regional_color_birth = True
            module.config.birth.regional_alpha_repair = True
            module.config.birth.residual_support_init = True
            module.config.birth.keep_incumbent_fixed_after_motion = True
            module.config.birth.keep_incumbent_alpha_after_motion = True
            module.config.birth.regional_color_min_support = 32.
            module.config.birth.regional_color_min_views = 2
            module.config.birth.regional_color_l1_threshold = .03
            module.config.rigid_tree.spatial_gate.final_joint_refine_steps = 0
            module._plan_regional_alpha_state = None
            checkpoint_harness(module)

        def evidence(proposal, cache):
            positions = proposal.gaussians.get_xyz.detach().cpu().numpy()
            den = np.full((len(cache), len(positions)), 40.)
            num = den * np.where(positions[:, 0] < .5, .1, .01)[None]
            return num, den, [dict(camera=x['name']) for x in cache]

        def geometry(raw, rows, cache, step):
            seed = {name:raw[name][rows].clone() for name in ('xyz','rotation','logscale')}
            return seed, torch.ones(len(rows), dtype=torch.bool), dict(policy='synthetic')

        with tempfile.TemporaryDirectory() as directory, patch.object(
                BirthGaussianModel, 'training_setup', cpu_gaussian_setup), \
                patch('pipeline.regional_topology.native_colour_evidence', side_effect=evidence), \
                patch('pipeline.frame_plan.residual_support_seed', side_effect=geometry):
            module, _ = controller_fixture(directory)
            configure(module)
            xyz = module.gaussians._xyz.detach()
            paths, parents, opened, centers, routing = spatial_layout(xyz, 5, return_centers=True,
                                                                       return_routing=True)
            layout = SimpleNamespace(spatial_paths=paths, spatial_centers=centers,
                **{k:routing.get(k) for k in ('spatial_support_ball','spatial_leaf_centers','spatial_leaf_nodes')})
            module._regional_color_layout = (layout, 'fixed fixture layout')
            module.finalize_frame = MethodType(Module.finalize_frame, module)
            prior, initial = copy.deepcopy(module.motion.frame_reference_state), model_raw(module.gaussians)
            writer = SimpleNamespace(output_dir=Path(directory), module=module, datamodule=module.datamodule)
            Trainer._write_reference(writer, 0, prior)
            module.trainer.decoder_reference_path = writer.decoder_reference_path
            runner = SimpleNamespace(module=module,
                config=SimpleNamespace(motion_views_per_step=4, birth_views_per_step=4),
                _views_per_step=lambda step:4)
            cleanup = module._plan_cleanup
            @torch.no_grad()
            def retire(step):
                if step == 250:
                    rows = module.opacity_update.rows
                    eligible = module._plan_regional_alpha_state['eligible'][rows]
                    row = rows[eligible][0]
                    module.opacity_update.delta[row] = -module.opacity_update.alpha_reference[row]
                return cleanup(step)
            module._plan_cleanup = retire
            clone = None
            for step in range(1, 301):
                module.trainer.global_step = step
                module.pre_training_step()
                Trainer._fit_views(runner, None, step)
                if step >= 200:
                    state = module._plan_regional_alpha_state
                    rows = module.opacity_update.rows
                    frozen = ~state['eligible'][rows]
                    self.assertTrue(torch.equal(module.opacity_update.delta[rows[frozen]],
                                                state['baseline_delta'][rows[frozen]]))
                if step == 250:
                    self.assertEqual(module._plan_events[-1]['removed_old'], 1)
                    packet = dict(global_step=250, current_frame=1)
                    Module.on_save_checkpoint(module, packet)
                    clone, _ = controller_fixture(directory)
                    configure(clone)
                    Module.on_load_checkpoint(clone, copy.deepcopy(packet))
                    clone.trainer.global_step = 251
                    clone.pre_training_step()
                    other = SimpleNamespace(module=clone, config=runner.config, _views_per_step=lambda step:4)
                    Trainer._fit_views(other, None, 251)
                if step == 251:
                    self.assertTrue(torch.equal(module.opacity_update.delta, clone.opacity_update.delta))
                    self.assertTrue(torch.equal(module.births.source_rows, clone.births.source_rows))
                    for name in RAW_FIELDS:
                        self.assertTrue(torch.equal(model_raw(module.births)[name], model_raw(clone.births)[name]))
            self.assertTrue(module.frame_finalized)
            self.assertGreater(module.frame_report['birth_count'], 0)
            decoded, raw = decode_frame(Path(directory)/'frame_storage'/'rigid_tree', 1, prior, initial)
            self.assertEqual(decoded.digest(), module.motion.codec_previous_state.digest())
            self.assertTrue(torch.equal(decoded.birth_sources, module.motion.pending_birth_sources))
            for name in RAW_FIELDS:
                self.assertTrue(torch.equal(raw[name], model_raw(module.gaussians)[name].cpu()), name)
            checks = module.birth_control['regional_alpha_repair']['invariant_checks']
            self.assertEqual([row['step'] for row in checks], [201, 250, 251, 300])
            for row in checks:
                self.assertEqual(row['max_frozen_alpha_change'], 0.)
                self.assertEqual(row['max_frozen_symbol_change'], 0.)
            Module.begin_next_frame(module)
            module.current_frame = 2
            Module.prepare_frame(module)
            self.assertIsNone(module._plan_regional_alpha_state)


class ReanchorAlphaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def fixture(self, directory, *, enabled=True, all_low=False):
        module, _ = controller_fixture(directory)
        module.config.birth.regional_color_birth = enabled
        module.config.birth.keep_incumbent_fixed_after_motion = True
        module.config.birth.keep_incumbent_alpha_after_motion = True
        module.motion_committed = True
        module.early_initialized = True
        module.births = FramePlanBirthModel(1, logit_opacity=True)
        module.births.spatial_lr_scale = 1.
        module._plan_events = []
        reference = module._adaptive_reference()
        reference['opacity'] = module.opacity_update.canonical_logits().to(module.device)
        # Source 1 can represent a value just above .01. Reanchoring the
        # same value to source 0 rounds it below .01 on the .01 logit grid.
        source_logit = reference['opacity'][1, 0]
        threshold = math.log(.01 / .99)
        k = math.ceil((threshold - float(source_logit)) / module.config.birth.new_opacity_step)
        low = source_logit + k * module.config.birth.new_opacity_step
        target = reference['opacity'][0, 0]
        reanchored = target + torch.round((low-target) / module.config.birth.new_opacity_step) * module.config.birth.new_opacity_step
        self.assertGreaterEqual(float(low.sigmoid()), .01)
        self.assertLess(float(reanchored.sigmoid()), .01)
        safe = torch.logit(torch.tensor(.1))
        for index, step in enumerate((200, 250)):
            raw = {name: value[[0, 0]].clone() for name, value in reference.items()}
            raw['opacity'][:, 0] = torch.tensor(([low, low] if all_low else
                                                  [low, safe] if index == 0 else [safe, low]))
            parents = torch.tensor([3 + 2*index, 4 + 2*index])
            module.births.add(raw, parents, torch.tensor([1, 1]), step,
                              module.motion.optimizer, module.config.gaussian_stage2)
        # Give every parameter identifiable row-wise state in both cohorts.
        # Subsequent selection must retain the right source row and moments.
        optimizer = module.motion.optimizer
        for cohort, batch in enumerate(module.births.cohorts):
            for parameter in batch['parameters'].values():
                shape = (2,) + (1,) * (parameter.ndim - 1)
                state_values = torch.arange(2, dtype=parameter.dtype).reshape(shape).expand_as(parameter) + 10*cohort
                optimizer.state[parameter] = dict(step=torch.tensor(7.),
                    exp_avg=state_values.clone(), exp_avg_sq=(state_values + 1).clone())
        return module

    def assert_survivors_and_state(self, module, step, old_removed=0):
        before = module.make_birth_view(reference=module._adaptive_reference()).get_opacity.flatten()
        self.assertTrue(bool((before >= .01).all()))
        old_count = len(module.gaussians._xyz)
        module._plan_cleanup(step)
        event = module._plan_events[-1]
        self.assertEqual(event['removed_old'], old_removed)
        self.assertEqual(event['removed_new'], 2)
        self.assertEqual(event['new_count_before_cleanup'], 4)
        self.assertEqual(event['new_count_before_reanchor'], 4)
        self.assertEqual(event['post_reanchor_low_alpha_pruned'], 2)
        self.assertEqual(event['new_count_after_reanchor'], 2)
        self.assertFalse(event['post_reanchor_alpha_check_deferred'])
        self.assertEqual(len(module.gaussians._xyz), old_count-old_removed)
        self.assertEqual(module.births.parent_rows.tolist(), [4, 5])
        self.assertEqual(module.births.source_rows.tolist(), [0, 0])
        self.assertEqual([len(c['parameters']['_xyz']) for c in module.births.cohorts], [1, 1])
        for cohort, batch in enumerate(module.births.cohorts):
            for name, parameter in batch['parameters'].items():
                state = module.motion.optimizer.state[parameter]
                self.assertEqual(int(state['step']), 7)
                expected = 1 if cohort == 0 else 10
                torch.testing.assert_close(state['exp_avg'], torch.full_like(parameter, expected))
                torch.testing.assert_close(state['exp_avg_sq'], torch.full_like(parameter, expected+1))
                group = next(g for g in module.motion.optimizer.param_groups
                             if g.get('name') == f'early_plan_{cohort}_{name}')
                self.assertIs(group['params'][0], parameter)
        view = module.make_birth_view(reference=module._adaptive_reference())
        self.assertTrue(bool((view.get_opacity.flatten() >= .01).all()))

    def test_reanchor_alpha_at_all_three_cleanup_boundaries(self):
        for step in (200, 250, 300):
            with self.subTest(step=step), tempfile.TemporaryDirectory() as directory:
                self.assert_survivors_and_state(self.fixture(directory), step)

    def test_no_surviving_newborns_is_safe(self):
        with tempfile.TemporaryDirectory() as directory:
            module = self.fixture(directory, all_low=True)
            module._plan_cleanup(300)
            self.assertEqual(module._plan_events[-1]['post_reanchor_low_alpha_pruned'], 4)
            self.assertEqual(module._plan_events[-1]['new_count_after_reanchor'], 0)
            self.assertIsNone(module.make_birth_view(reference=module._adaptive_reference()))
            self.assertEqual(module.births.source_rows.numel(), 0)
            self.assertEqual([len(batch['parameters']['_xyz']) for batch in module.births.cohorts], [0, 0])

    def test_counts_separate_initial_and_post_reanchor_pruning(self):
        with tempfile.TemporaryDirectory() as directory:
            module = self.fixture(directory)
            with torch.no_grad():
                module.births.cohorts[0]['parameters']['_opacity'][0] = torch.logit(torch.tensor(.005))
            module._plan_cleanup(250)
            event = module._plan_events[-1]
            self.assertEqual(event['new_count_before_cleanup'], 4)
            self.assertEqual(event['new_count_before_reanchor'], 3)
            self.assertEqual(event['post_reanchor_low_alpha_pruned'], 1)
            self.assertEqual(event['new_count_after_reanchor'], 2)
            self.assertEqual(event['removed_new'], 2)
            self.assertEqual(module.births.parent_rows.tolist(), [4, 5])

    def test_default_off_keeps_existing_reanchor_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            module = self.fixture(directory, enabled=False)
            module._plan_cleanup(300)
            self.assertEqual(len(module.births._xyz), 4)
            alpha = module.make_birth_view(reference=module._adaptive_reference()).get_opacity.flatten()
            self.assertEqual(int((alpha < .01).sum()), 2)
            self.assertNotIn('post_reanchor_low_alpha_pruned', module._plan_events[-1])

    def test_final_encoded_and_decoded_births_match_filtered_state(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(
                BirthGaussianModel, 'training_setup', cpu_gaussian_setup):
            module = self.fixture(directory)
            checkpoint_harness(module)
            prior = copy.deepcopy(module.motion.frame_reference_state)
            initial = model_raw(module.gaussians)
            writer = SimpleNamespace(output_dir=Path(directory), module=module, datamodule=module.datamodule)
            Trainer._write_reference(writer, 0, prior)
            module.trainer.decoder_reference_path = writer.decoder_reference_path
            module.trainer.global_step = module.early_shared_steps = 300
            module.birth_control['early_replacement'] = dict(actual_birth_cohorts=2, source_count=4)
            # Retire a real old GS too, so the motion packet survivor mapping
            # and the rebased birth-source IDs are independently decoded.
            index = module.opacity_update.rows[-1]
            with torch.no_grad():
                module.opacity_update.delta[index] = -module.opacity_update.alpha_reference[index]
            self.assert_survivors_and_state(module, 300, old_removed=1)
            reference = module._adaptive_reference()
            expected, sources = materialize_quantized_gaussians(reference,
                module.make_birth_view(reference=reference))
            Module.finalize_frame(module)
            stream_dir = Path(directory) / 'frame_storage' / 'rigid_tree'
            state, decoded = decode_frame(stream_dir, 1, prior, initial)
            torch.testing.assert_close(state.birth_sources, sources.cpu(), rtol=0, atol=0)
            self.assertEqual(len(sources), 2)
            self.assertEqual(module.frame_report['birth_count'], 2)
            self.assertTrue(bool((decoded['opacity'][-2:].sigmoid() >= .01).all()))
            for name in RAW_FIELDS:
                torch.testing.assert_close(decoded[name], expected[name].cpu(), rtol=0, atol=0)


# ----- Residual support geometry -----


from torch import nn

from pipeline.model.birth_quantization import RAW_ATTRIBUTES, RAW_FIELDS, model_raw
from pipeline.model.gaussian import GaussianModel
from pipeline.model.residual_support_geometry import residual_support_seed, residual_support_opacity, support_distribution, _matrix_quaternion
from pipeline.model.support_reincarnation_proposal import _native_responsibility, _rotation, score_support_responsibility


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


class LinearNativeContractRenderer:
    """CPU renderer with the exact colour Jacobian sum_i Talpha_i*c_i."""
    def __init__(self,weights):
        self.weights=weights;self.gaussians=object();self.collect_contributions=False
        self.zero_forward_values=[];self.fail_zero=False

    def __call__(self,view,background,override_color=None):
        if override_color is None:
            image=view.rendered
        else:
            if self.fail_zero:raise RuntimeError('intentional zero render failure')
            self.assert_zero=bool(override_color.detach().eq(0).all())
            image=torch.einsum('nhw,nc->chw',self.weights,override_color)
            self.zero_forward_values.append(float(image.detach().abs().max()))
        result={'render':image}
        if self.collect_contributions:
            result['motion_contribution_filter']=self.weights.sum((1,2))>0
        return result


def support_fixture(constant=False):
    model=GaussianModel(3);model.active_sh_degree=3
    raw=dict(xyz=torch.tensor([[0.,0.,3.]]).repeat(4,1),
        rotation=torch.tensor([[2.,.1,-.2,.3]]).repeat(4,1),
        logscale=torch.tensor([[4.,2.,.5],[.1,.08,.001],[.5,.3,.01],[.6,.4,.002]]).log(),
        opacity=torch.tensor([[.7],[-100.],[.2],[.3]]),
        sh_dc=torch.arange(12.).reshape(4,1,3)/10,
        sh_rest=torch.arange(4*15*3.).reshape(4,15,3)/100)
    for name,attr in zip(RAW_FIELDS,RAW_ATTRIBUTES):setattr(model,attr,nn.Parameter(raw[name].clone()))
    weights=torch.tensor([.3,0.,.1,.2])[:,None,None].expand(4,8,8).clone()
    renderer=LinearNativeContractRenderer(weights)
    xx=torch.arange(8).float()[None].expand(8,-1)
    residual=torch.full((8,8),.1) if constant else (xx-3.5)*.05
    gt=torch.full((3,8,8),.4)
    views=[SimpleNamespace(path=f'cam{i:02d}.png',image=gt.clone(),rendered=gt+residual,
        fovx=2*math.atan(.5),fovy=2*math.atan(.5),world_view_transform=torch.eye(4)) for i in (1,2,3)]
    module=SimpleNamespace(gaussians=model,renderer=renderer,background=torch.zeros(3))
    return module,views,raw,weights



class SupportResponsibilityTests(unittest.TestCase):
    def test_zero_forward_retains_true_positive_SSE_colour_gradient(self):
        module,views,_,weights=support_fixture()
        scores,observed,bits,report=score_support_responsibility(module,views)
        error=sum((v.rendered-v.image).square().sum(0) for v in views)
        expected=(weights*error).sum((1,2))
        torch.testing.assert_close(scores,expected,rtol=1e-6,atol=1e-8)
        self.assertEqual(module.renderer.zero_forward_values,[0.,0.,0.])
        torch.testing.assert_close(observed,torch.tensor([7,0,7,7]))
        torch.testing.assert_close(bits,torch.tensor([7,0,7,7]))
        self.assertAlmostEqual(report['native_responsibility_fraction_of_total_SSE'],.6,places=6)
        for attr in RAW_ATTRIBUTES:self.assertIsNone(getattr(module.gaussians,attr).grad)

    def test_opposite_residual_signs_cancel_DC_but_not_responsibility(self):
        module,views,_,weights=support_fixture()
        signed=(weights[0]*(views[0].rendered-views[0].image)).sum()
        self.assertLess(abs(float(signed)),1e-6)
        scores,_,_,_=score_support_responsibility(module,views)
        self.assertGreater(float(scores[0]),0)

    def test_renderer_flags_and_model_restore_even_on_exception(self):
        module,views,_,_=support_fixture();original=module.renderer.gaussians
        module.renderer.collect_contributions=True
        score_support_responsibility(module,views)
        self.assertIs(module.renderer.gaussians,original)
        self.assertTrue(module.renderer.collect_contributions)
        module.renderer.fail_zero=True
        with self.assertRaisesRegex(RuntimeError,'intentional zero'):
            score_support_responsibility(module,views)
        self.assertIs(module.renderer.gaussians,original)
        self.assertTrue(module.renderer.collect_contributions)

    def test_test_camera_and_nonmaterialized_overrides_are_rejected(self):
        module,views,_,_=support_fixture();views[0].path='cam00.png'
        with self.assertRaisesRegex(ValueError,'external test'):score_support_responsibility(module,views)
        views[0].path='cam01.png';module.gaussians.rigid_alpha=torch.ones(4,1)
        with self.assertRaisesRegex(ValueError,'materialized'):score_support_responsibility(module,views)


if __name__ == "__main__":
    unittest.main()
