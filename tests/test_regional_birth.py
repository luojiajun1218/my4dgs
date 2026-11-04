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
from tests.test_residual_support_geometry import support_fixture


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
            for step in range(1, 201):
                case.step(m, trainer, step)
            names = list(m._plan_admission_camera_names)
            self.assertFalse(set(names) & set(m._plan_events[0]['proposal_cameras']))
            checkpoint = dict(global_step=200, current_frame=1)
            Module.on_save_checkpoint(m, checkpoint)
            resumed, resumed_trainer, _, _ = case.setup_controller(resume_dir, True)
            resumed.config.birth.regional_postfit_admission = True
            Module.on_load_checkpoint(resumed, copy.deepcopy(checkpoint))
            self.assertEqual(resumed._plan_admission_camera_names, names)
            for step in range(201, 300):
                case.step(m, trainer, step)
                self.assertFalse(set(names) & {Path(v.path).name for v in m._plan_step_views})
                if step in (201, 250):
                    self.assertEqual([v.path for v in m._plan_step_views],
                        [v.path for v in resumed.frame_plan_training_views(step, 4)])
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


if __name__ == "__main__":
    unittest.main()
