"""Birth tests."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import importlib
import math
import unittest
from tempfile import TemporaryDirectory
from types import SimpleNamespace, SimpleNamespace as NS
from unittest.mock import Mock, patch

import torch

from pipeline.model import residual_birth as rb
from pipeline.model.birth_quantization import (
    RAW_ATTRIBUTES,
    RAW_FIELDS,
    materialize_quantized_gaussians,
    model_raw,
)
from pipeline.model.child_dc_initializer import (
    _pair_image_moments,
    initialize_child_dc_once,
    solve_box2,
)
from pipeline.model.gaussian import BirthGaussianModel
from pipeline.model.rigid_residual_tree import RigidTreeConfig
from pipeline.model.spatial_gated_motion import SpatialGatedMotion

# ----- test_residual_birth -----
def camera(x, width=96, height=80, focal=80.):
    matrix = torch.eye(4)
    matrix[3, 0] = -x
    u, v = torch.meshgrid(torch.arange(width), torch.arange(height), indexing='xy')
    world_x = (u+.5-width/2)/focal*3+x
    world_y = (v+.5-height/2)/focal*3
    image = torch.stack((.5+.2*torch.sin(9*world_x+5*world_y),
                         .5+.25*torch.cos(7*world_x-11*world_y),
                         .5+.2*torch.sin(13*world_x+8*world_y)))
    return NS(image=image, fovx=2*math.atan(width/(2*focal)),
              fovy=2*math.atan(height/(2*focal)),
              world_view_transform=matrix, camera_center=torch.tensor([x, 0., 0.]))


class GeometryTests(unittest.TestCase):
    def test_projection_and_inverse_use_pixel_centers(self):
        view = camera(.3)
        pixels = torch.tensor([[0., 0.], [47., 39.], [95., 79.]])
        depths = torch.tensor([2., 3., 4.])
        xyz = rb.unproject(view, pixels, depths)
        grid, recovered = rb.project(view, xyz)
        expected = (2*pixels+1)/torch.tensor([96., 80.])-1
        torch.testing.assert_close(grid, expected)
        torch.testing.assert_close(recovered, depths)

    def test_rgb_search_recovers_surface_without_incumbent_depth(self):
        reference = camera(0.)
        peers = [camera(-.3), camera(.2), camera(.5)]
        pixels = torch.tensor([[40., 32.], [55., 45.], [61., 25.]])
        xyz, _, depth, _, cost, valid = rb.match_depth(reference, peers, pixels,
                                                      1., 6., 64)
        self.assertTrue(bool(valid.all()))
        torch.testing.assert_close(depth, torch.full((3,), 3.), atol=.025, rtol=0)
        self.assertLess(float(cost.max()), .03)
        # A wrong old surface at z=5 was not used or copied into these births.
        self.assertGreater(float((xyz[:, 2]-5).abs().min()), 1.9)

    def test_quaternion_handles_half_turn_and_groups_cover_once(self):
        q = rb.matrix_quaternion(torch.diag(torch.tensor([-1., 1., -1.])))
        torch.testing.assert_close(q.abs(), torch.tensor([0., 0., 1., 0.]))
        xyz = torch.randn(23, 3, generator=torch.Generator().manual_seed(7))
        groups = rb.spatial_groups(xyz, 8)
        self.assertEqual(len(groups), 8)
        self.assertEqual(sorted(torch.cat(groups).tolist()), list(range(23)))
        self.assertEqual(rb.spatial_groups(xyz[:0], 8), [])

    def test_flat_chromatic_patch_is_not_geometric_evidence(self):
        views = [camera(x) for x in (0., -.3, .2, .5)]
        for view in views:
            view.image[:] = torch.tensor([.7, .3, .3])[:, None, None]
        *_, valid = rb.match_depth(views[0], views[1:], torch.tensor([[40., 32.]]),
                                   1., 6., 64)
        self.assertFalse(bool(valid.any()))

    def test_region_allocation_preserves_integrated_residual_mass(self):
        residual = torch.zeros(96, 128)
        residual[16:80, 16:112] = .1
        image = torch.full((3, 96, 128), .5)
        regions = rb.residual_regions(residual, image, 12, .03)
        self.assertEqual(len(regions), 12)
        self.assertAlmostEqual(sum(region['total_mass'] for region in regions),
                               float(residual.square().sum()), places=4)
        self.assertEqual(sum(region['total_area'] for region in regions), 64*96)
        self.assertTrue(all(len(region['pixels']) >= 4 for region in regions))

    def patch_fixture(self, span=20.):
        view = camera(0.)
        offsets = torch.tensor([[x, y] for y in (-1., 0., 1.) for x in (-1., 0., 1.)])
        pixels = torch.tensor([48., 40.])+offsets*span
        depth = torch.full((9,), 3.)
        xyz = rb.unproject(view, pixels, depth)
        region = dict(pixels=pixels, mass=torch.ones(9), area=torch.full((9,), span**2))
        return view, region, xyz, torch.full((9, 3), .5), depth, torch.full((9,), .001), torch.zeros(9), torch.ones(9, dtype=torch.bool)

    def test_surface_covariance_follows_observed_region_support(self):
        small = rb.region_surface_patches(*self.patch_fixture(4.))
        large = rb.region_surface_patches(*self.patch_fixture(20.))
        self.assertEqual(len(small), 1)
        self.assertEqual(len(large), 1)
        self.assertGreater(large[0]['diagnostic']['projected_integral_area'],
                           small[0]['diagnostic']['projected_integral_area']*20)
        self.assertEqual(large[0]['diagnostic']['matched_anchors'], 9)

    def test_nonplanar_matches_cannot_become_a_large_surface_blob(self):
        args = list(self.patch_fixture())
        args[4] = args[4]+torch.tensor([.2, -.2, .2, -.2, .2, -.2, .2, -.2, .2])
        args[2] = rb.unproject(args[0], args[1]['pixels'], args[4])
        self.assertEqual(rb.region_surface_patches(*args), [])

    def test_appearance_discontinuity_does_not_create_a_shared_surface(self):
        args = list(self.patch_fixture())
        args[3][:4] = torch.tensor([.9, .1, .1])
        args[3][4:] = torch.tensor([.1, .1, .9])
        patches = rb.region_surface_patches(*args)
        self.assertTrue(all(patch['diagnostic']['matched_anchors'] < 9 for patch in patches))
        self.assertTrue(all(patch['color'].max() > .8 for patch in patches))

    def test_region_lineage_survives_admission_subdivision(self):
        transaction = NS(group_ids=torch.tensor([0, 1, 0, 2]))
        xyz = torch.tensor([[0., 0., 0.], [1., 0., 0.], [100., 0., 0.], [101., 0., 0.]])
        groups = rb.lineage_groups(transaction, torch.arange(4), xyz, 2)
        self.assertTrue(any(0 in group and 2 in group for group in groups))

    def test_proposal_builds_finite_region_patches_with_aligned_lineage(self):
        views = [camera(x) for x in (0., -.3, .2)]
        for i, view in enumerate(views):
            view.path = f'fit{i}.png'; view.image.fill_(.5)
        base = NS(_xyz=torch.tensor([[0., 0., 3.]]), _rotation=torch.tensor([[1., 0., 0., 0.]]),
                  _features_dc=torch.zeros(1, 1, 3), _features_rest=torch.zeros(1, 3, 3),
                  _scaling=torch.full((1, 3), -.7), _opacity=torch.ones(1, 1))
        module = NS(device=torch.device('cpu'), gaussians=base,
                    datamodule=NS(cameras_extent=3.), background=torch.zeros(3))
        module.renderer = lambda view, bg: {'render': view.image+.1+base.rigid_sh_dc.mean()*.01}
        def match(view, peers, pixels, *unused):
            depth = torch.full((len(pixels),), 3.)
            return (rb.unproject(view, pixels, depth), torch.full((len(pixels), 3), .5),
                    depth, torch.full_like(depth, .001), torch.zeros_like(depth),
                    torch.ones(len(pixels), dtype=torch.bool))
        config = rb.ResidualBirthConfig(max_candidates=6, repair_candidates=0, proposal_views=3)
        with patch.object(rb, 'match_depth', match):
            proposal = rb.propose(module, config, views, tuple(view.path for view in views), ())
        self.assertGreater(proposal.proposed_spatial, 0)
        self.assertLessEqual(len(proposal.sources), 6)
        self.assertEqual(len(proposal.group_ids), len(proposal.sources))
        self.assertTrue(bool(torch.isfinite(proposal.raw['logscale']).all()))
        self.assertGreater(proposal.region_diagnostics['matched_anchor_assigned_region_pixels'], 0)


class AdmissionTests(unittest.TestCase):
    def fixture(self, count=2):
        xyz = torch.arange(max(count, 1)*3).reshape(-1, 3).float()[:count]
        transaction = NS(observations=torch.full((count,), 3, dtype=torch.long),
            fit_names=('fit1', 'fit2'), validation_names=('inner1', 'inner2'),
            replacements=torch.tensor([0, -1])[:count],
            scores=torch.tensor([1., .5])[:count], proposed_spatial=count,
            proposed_repairs=0)
        return NS(device=torch.device('cpu'), current_frame=1,
            birth_transaction=transaction, births=NS(_xyz=xyz, _opacity=torch.zeros(count, 1)),
            gaussians=NS(_xyz=torch.zeros(4, 3), _opacity=torch.full((4, 1), 2.)),
            config=NS(opacity_update=NS(prune_threshold=.01), birth=NS(opacity_threshold=.01)),
            motion=NS(observations=torch.ones(4, dtype=torch.long)))

    @staticmethod
    def encode(module, mask, delete, **unused):
        count = int((~delete).sum()+mask.sum())
        return rb.CommitTrial((~delete).nonzero(as_tuple=True)[0],
            mask.nonzero(as_tuple=True)[0], torch.zeros(int(mask.sum()), dtype=torch.long),
            dict(mask=mask.clone(), delete=delete.clone(),
                 xyz=torch.zeros(count, 3), rotation=torch.zeros(count, 4)), 1000+20*int(mask.sum()),
            1000, 20*int(mask.sum()), 0)

    def test_harmful_replacement_rolls_back_parent_and_good_birth_can_commit(self):
        module = self.fixture()
        original = module.gaussians._opacity.clone()
        def errors(module, views, raw):
            if bool(raw['mask'][0]):
                return torch.tensor([1.4, .8])  # Aggregate distortion is worse.
            return torch.tensor([.9, .9]) if bool(raw['mask'][1]) else torch.ones(2)
        with patch.object(rb, 'encode_trial', self.encode), \
                patch.object(rb, 'validation_errors', errors):
            trial, report = rb.admit(module, rb.ResidualBirthConfig(max_groups=2), [])
        self.assertEqual(trial.candidates.tolist(), [1])
        self.assertEqual(trial.keep.tolist(), [0, 1, 2, 3])
        self.assertEqual(report['accepted'], 1)
        torch.testing.assert_close(module.gaussians._opacity, original)

    def test_low_alpha_does_not_authorize_losing_unique_coverage(self):
        module = self.fixture(0)
        module.gaussians._opacity[0] = -8.
        def errors(module, views, raw):
            return torch.tensor([1.5, .8]) if bool(raw['delete'][0]) else torch.ones(2)
        with patch.object(rb, 'encode_trial', self.encode), \
                patch.object(rb, 'validation_errors', errors):
            trial, report = rb.admit(module, rb.ResidualBirthConfig(), [])
        self.assertEqual(len(trial.keep), 4)
        self.assertEqual(report['retired'], 0)

    def test_bad_candidate_cannot_veto_an_entire_mixed_group(self):
        module = self.fixture()
        module.birth_transaction.replacements.fill_(-1)
        def errors(module, views, raw):
            if bool(raw['mask'][0]):
                return torch.tensor([1.4, .8])
            return torch.tensor([.9, .9]) if bool(raw['mask'][1]) else torch.ones(2)
        with patch.object(rb, 'encode_trial', self.encode), \
                patch.object(rb, 'validation_errors', errors):
            trial, report = rb.admit(module, rb.ResidualBirthConfig(max_groups=2), [])
        self.assertEqual(trial.candidates.tolist(), [1])
        self.assertGreater(report['candidate_groups_evaluated'], 2)

    def test_unobserved_low_alpha_is_dormant_not_automatically_dead(self):
        module = self.fixture(0)
        module.gaussians._opacity[0] = -8.
        module.motion.observations[0] = 0
        with patch.object(rb, 'encode_trial', self.encode), \
                patch.object(rb, 'validation_errors', lambda *args: torch.ones(2)):
            trial, report = rb.admit(module, rb.ResidualBirthConfig(), [])
        self.assertEqual(len(trial.keep), 4)
        self.assertEqual(report['tested_low_alpha_deaths'], 0)

    def test_canonical_zero_can_retire_without_observations(self):
        module = self.fixture(0)
        module.gaussians._opacity[0] = -100.
        module.motion.observations.zero_()
        with patch.object(rb, 'encode_trial', self.encode), \
                patch.object(rb, 'validation_errors', lambda *args: torch.ones(2)):
            trial, report = rb.admit(module, rb.ResidualBirthConfig(), [])
        self.assertEqual(trial.keep.tolist(), [1, 2, 3])
        self.assertEqual(report['retired'], 1)

    def test_low_alpha_retirement_allows_lower_mean_with_a_camera_tradeoff(self):
        module = self.fixture(0)
        module.gaussians._opacity[0] = -8.
        def errors(module, views, raw):
            return torch.tensor([1.1, .7]) if bool(raw['delete'][0]) else torch.ones(2)
        with patch.object(rb, 'encode_trial', self.encode), \
                patch.object(rb, 'validation_errors', errors):
            trial, report = rb.admit(module, rb.ResidualBirthConfig(), [])
        self.assertEqual(trial.keep.tolist(), [1, 2, 3])
        self.assertEqual(report['retired'], 1)

    def test_only_one_observed_fitting_camera_cannot_admit_a_birth(self):
        module = self.fixture()
        module.birth_transaction.observations.fill_(1)
        with patch.object(rb, 'encode_trial', self.encode), \
                patch.object(rb, 'validation_errors', lambda *args: torch.ones(2)):
            trial, report = rb.admit(module, rb.ResidualBirthConfig(), [])
        self.assertEqual(len(trial.candidates), 0)
        self.assertEqual(report['observed_candidates'], 0)

    def test_repair_source_deletion_preserves_coverage_outside_acceptance_views(self):
        module = self.fixture(1)
        validation, retirement = ['inner'], ['inner', 'fitting']
        def errors(module, views, raw):
            if not bool(raw['mask'][0]):
                return torch.ones(len(views))
            if views is validation:
                return torch.tensor([.8])
            # The replacement improves the acceptance view but loses source
            # coverage in a fitting view, hidden from the birth validation set.
            return torch.tensor([.8, 1.4]) if bool(raw['delete'][0]) else torch.tensor([.9, .9])
        with patch.object(rb, 'encode_trial', self.encode), \
                patch.object(rb, 'validation_errors', errors):
            trial, report = rb.admit(module, rb.ResidualBirthConfig(), validation, retirement)
        self.assertEqual(trial.candidates.tolist(), [])
        self.assertEqual(trial.keep.tolist(), [0, 1, 2, 3])
        self.assertEqual(report['rejected_repair_source_deaths'], 1)
        self.assertEqual(report['retirement_committed_mse'], [1., 1.])

    def test_repair_guard_compares_atomic_replacement_with_incumbent(self):
        module = self.fixture(1)
        validation, retirement = ['inner'], ['inner', 'fitting']
        def errors(module, views, raw):
            if not bool(raw['mask'][0]):
                return torch.ones(len(views))
            if views is validation:
                return torch.tensor([.8])
            # Replacement improves the real incumbent in both views. A scene
            # with old+new splats would have different double-opacity coverage
            # and must not be the guard's comparison baseline.
            return torch.tensor([.8, .95]) if bool(raw['delete'][0]) else torch.tensor([.9, .85])
        with patch.object(rb, 'encode_trial', self.encode), \
                patch.object(rb, 'validation_errors', errors):
            trial, report = rb.admit(module, rb.ResidualBirthConfig(), validation, retirement)
        self.assertEqual(trial.candidates.tolist(), [0])
        self.assertEqual(trial.keep.tolist(), [1, 2, 3])
        self.assertEqual(report['tested_repair_source_deaths'], 1)
        torch.testing.assert_close(torch.tensor(report['retirement_committed_mse']), torch.tensor([.8, .95]))


# ----- test_birth_source_covariance -----
GAUSSIAN = importlib.import_module('pipeline.model.gaussian')


class BirthSourceCovarianceTests(unittest.TestCase):
    def test_clone_returns_exact_sources_with_budget_ties_and_rejected_rows(self):
        for budget, expected in ((None, [1, 2, 5]), (0, []), (1, [1]),
                                 (2, [1, 2]), (5, [1, 2, 5])):
            with self.subTest(budget=budget):
                rows = torch.arange(6, dtype=torch.float32)
                scale = torch.tensor([[.1]] * 3 + [[1.], [.1], [.1]]).expand(-1, 3)
                source = SimpleNamespace(_xyz=rows[:, None].expand(-1, 3),
                    xyz_gradient_accum=torch.tensor([[.2], [.8], [.8], [.9],
                                                    [float('nan')], [.7]]),
                    denom=torch.ones(6, 1), get_scaling=scale, percent_dense=.5,
                    _rotation=torch.tensor([[1., 0., 0., 0.]] * 6),
                    _scaling=scale.log(), _opacity=rows[:, None],
                    _features_dc=rows[:, None, None].expand(-1, 1, 3),
                    _features_rest=rows[:, None, None].expand(-1, 3, 3))
                captured = []
                source.densification_postfix = lambda *values: captured.append(values)
                with patch('torch.normal', side_effect=lambda mean, std: mean.clone()):
                    report, selected = GAUSSIAN.GaussianModel.adaptive_densify_and_clone(
                        source, .5, 1., max_new_gaussians=budget, return_source_rows=True)
                torch.testing.assert_close(selected, torch.tensor(expected, dtype=torch.long))
                self.assertEqual(report, dict(eligible=3, added=len(expected),
                                              blocked=3 - len(expected)))
                # Geometry and appearance rows must agree with the returned lineage.
                torch.testing.assert_close(captured[0][0], source._xyz[selected])
                torch.testing.assert_close(captured[0][1], source._features_dc[selected])
                torch.testing.assert_close(captured[0][2], source._features_rest[selected])

    @unittest.skipUnless(torch.cuda.is_available(), 'Actual CUDA newborn splits')
    def test_split_budget_preserves_child_order_and_source_lineage(self):
        fields = ('_xyz', '_features_dc', '_features_rest', '_opacity',
                  '_scaling', '_rotation')
        def prune(model, remove):
            for field in fields:
                setattr(model, field, getattr(model, field)[~remove])

        cases = ((2, None, [1, 2, 3, 4]), (2, 6, []), (2, 7, [1]),
                 (2, 9, [1, 2, 3]), (3, 8, [1]))
        for children, budget, selected_rows in cases:
            with self.subTest(children=children, budget=budget):
                model = GAUSSIAN.BirthGaussianModel(1)
                ids = torch.tensor([11, 22, 33, 44, 55, 66], device='cuda')
                model._xyz = ids[:, None].expand(-1, 3).float()
                model._features_dc = ids[:, None, None].expand(-1, 1, 3).float()
                model._features_rest = ids[:, None, None].expand(-1, 3, 3).float()
                model._opacity = ids[:, None].float()
                model._rotation = model._xyz.new_tensor([[1., 0., 0., 0.]] * 6)
                model._scaling = model._xyz.new_tensor(
                    [[.1], [.5], [.6], [.3], [.4], [.5]]).expand(-1, 3).log()
                model.percent_dense = .2
                model.source_rows = ids.clone()
                def postfix(*values):
                    for field, value in zip(fields, values):
                        setattr(model, field, torch.cat((getattr(model, field), value)))
                model.densification_postfix = postfix
                grads = model._xyz.new_tensor([[.9], [.8], [.8], [.7], [.6], [.4]])
                with patch.object(GAUSSIAN.BaseGaussianModel, 'prune_points', new=prune), \
                        patch('torch.normal', side_effect=lambda mean, std: torch.zeros_like(mean)):
                    report = model.adaptive_densify_and_split(
                        grads, .5, 1., children, max_points=budget)
                selected = torch.tensor(selected_rows, device='cuda', dtype=torch.long)
                survivors = torch.tensor([i for i in range(6) if i not in selected_rows],
                                         device='cuda', dtype=torch.long)
                expected = torch.cat((ids[survivors], ids[selected].repeat(children)))
                torch.testing.assert_close(model.source_rows, expected)
                torch.testing.assert_close(model._xyz[:, 0], expected.float())
                torch.testing.assert_close(model._features_dc[:, 0, 0], expected.float())
                self.assertEqual(report, dict(eligible=4,
                    added=len(selected_rows) * (children - 1), blocked=4 - len(selected_rows)))

    def test_frozen_newborn_orientation_inherits_raw_source_without_gradient(self):
        from pipeline.module import BirthConfig
        from pipeline.model.gaussian import GaussianModel
        from pipeline.model.birth_quantization import (
            QuantizedNewGaussianView, RAW_ATTRIBUTES, materialize_quantized_gaussians)
        import test_codec as replacement
        reference = replacement.reference_raw()
        reference['rotation'][1] = torch.tensor([2., .3, -.2, .1])
        source = torch.tensor([1, 3])
        model = GaussianModel(1)
        for name, attr in zip(replacement.RAW_FIELDS, RAW_ATTRIBUTES):
            setattr(model, attr, torch.nn.Parameter(reference[name][source].clone()))
        with torch.no_grad():
            model._rotation.fill_(.2)
        view = QuantizedNewGaussianView(model, BirthConfig(rotation_updates=False),
                                        2., reference, source)
        (view.get_xyz.sum() + view.get_rotation.sum()).backward()
        self.assertIsNone(model._rotation.grad)
        self.assertIsNotNone(model._xyz.grad)
        self.assertTrue(bool(view.symbols['rotation'].eq(0).all()))
        raw, _ = materialize_quantized_gaussians(reference, view)
        torch.testing.assert_close(raw['rotation'][-2:].view(torch.int32),
                                   reference['rotation'][source].view(torch.int32), rtol=0, atol=0)

    def test_rotated_source_controls_sampling_and_newborn_orientation(self):
        captured = {}
        q = torch.tensor([[2 ** -.5, 0., 0., 2 ** -.5]])
        scale = torch.tensor([[.1, .01, .001]])
        source = SimpleNamespace(_xyz=torch.zeros(1, 3), xyz_gradient_accum=torch.ones(1, 1),
            denom=torch.ones(1, 1), get_scaling=scale, percent_dense=.5,
            _rotation=q, _scaling=scale.log(), _opacity=torch.zeros(1, 1),
            _features_dc=torch.ones(1, 1, 3), _features_rest=torch.zeros(1, 3, 3))

        def postfix(*values):
            captured['values'] = values

        source.densification_postfix = postfix
        with patch('torch.normal', return_value=torch.tensor([[.2, 0., 0.]])):
            result = GAUSSIAN.GaussianModel.adaptive_densify_and_clone(
                source, .1, 1., sigma_scale=2., source_covariance_init=True)
        self.assertEqual(result['added'], 1)
        xyz, dc, rest, opacity, logscale, rotation = captured['values']
        torch.testing.assert_close(xyz, torch.tensor([[0., .2, 0.]]), rtol=0, atol=1e-7)
        torch.testing.assert_close(rotation, q, rtol=0, atol=0)
        torch.testing.assert_close(logscale, scale.log(), rtol=0, atol=0)
        torch.testing.assert_close(source._xyz, torch.zeros(1, 3), rtol=0, atol=0)
        torch.testing.assert_close(source._opacity, torch.zeros(1, 1), rtol=0, atol=0)

    def test_refinement_birth_allocation_checkpoint_cannot_silently_resume(self):
        from unittest.mock import Mock
        module = importlib.import_module('pipeline.module')
        harness = SimpleNamespace(gaussians=SimpleNamespace(restore=Mock()),
            renderer=SimpleNamespace(), config=SimpleNamespace(gaussian=object(),
                birth=module.BirthConfig()))
        with self.assertRaisesRegex(ValueError, 'newborn allocation.*replay-only'):
            module.Module.on_load_checkpoint(harness, dict(gaussians=object(),
                rigid_tree_version=1, current_frame=1,
                birth_allocation='clone_small_refine_large_v1'))


# ----- test_birth_cohorts -----
MODULE = importlib.import_module('pipeline.module')
TRAINER = importlib.import_module('pipeline.trainer')


def raw_point():
    return dict(xyz=torch.zeros(1, 3), rotation=torch.tensor([[1., 0., 0., 0.]]),
                logscale=torch.zeros(1, 3), opacity=torch.zeros(1, 1),
                sh_dc=torch.zeros(1, 1, 3), sh_rest=torch.zeros(1, 3, 3))


class CohortTests(unittest.TestCase):
    def test_trainer_rechecks_extended_deadline_and_resumes_the_second_cohort(self):
        class Progress:
            def __init__(self, **kwargs):
                self.n, self.total = kwargs['initial'], kwargs['total']
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def update(self, count): self.n += count
            def set_postfix(self, *args, **kwargs): pass
        with TemporaryDirectory() as directory:
            config = NS(rigid_tree=NS(spatial_gate=NS(gate_start_step=0, freeze_step=0),
                                     amplitude_refit_steps=0))
            module = NS(config=config, motion_steps=1, total_steps=3, frame_report={},
                        stage_image_metrics=[], pre_training_step=Mock())
            trainer = NS(current_frame=1, global_step=1, num_steps=3, module=module,
                config=NS(stage_image_camera=None), output_dir=Path(directory),
                _make_fetcher=lambda: object(), _resume_fetcher=Mock(),
                _materialize_step_metrics=TRAINER.Trainer._materialize_step_metrics,
                validation_loop=lambda: {}, ckpt_save_path=Path(directory)/'checkpoint.pth')
            executed, saved, bars = [], [], []
            def fit(fetcher, step, **kwargs):
                executed.append(step)
                if step == 3:
                    trainer.num_steps = module.total_steps = 6
                return torch.tensor(1.), dict(image_loss=torch.tensor(1.), psnr=torch.tensor(1.), gs=1)
            trainer._fit_views = fit
            trainer._checkpoint_due = lambda: trainer.global_step in (3, 6)
            trainer.save_checkpoint = lambda: saved.append(trainer.global_step)
            def progress(**kwargs):
                bar = Progress(**kwargs); bars.append(bar); return bar
            with patch.object(TRAINER, 'tqdm', progress):
                report = TRAINER.Trainer.frame_training_loop(trainer)
            self.assertEqual(executed, [2, 3, 4, 5, 6])
            self.assertEqual(saved, [3, 6])
            self.assertEqual(report['train_loss_steps'], 5)
            self.assertEqual(bars[0].total, 6)
            self.assertEqual(bars[0].n, 6)

    def test_repair_subkernels_are_deterministic_and_preserve_geometry_moments(self):
        raw = raw_point()
        domain = rb.covariance_support(raw, 0)
        color = torch.tensor([[.9, .1, .1] if point[0] < 0 else [.1, .1, .9]
                              for point in domain['xyz']])
        evidence = dict(valid=torch.ones(9, dtype=torch.bool), error=torch.ones(9), color=color)
        first = rb.split_support(raw, 0, domain, evidence)
        second = rb.split_support(raw, 0, domain, evidence)
        self.assertIsNotNone(first)
        children, domains, _ = first
        self.assertEqual(len(children), 2)
        weights = torch.tensor([len(item['xyz']) for item in domains])/9
        center = sum(weight*child['xyz'] for weight, child in zip(weights, children))
        covariance = sum(weight*(rb.raw_covariance({key: value[None] for key, value in child.items()}, 0)
                                + torch.outer(child['xyz']-center, child['xyz']-center))
                         for weight, child in zip(weights, children))
        torch.testing.assert_close(center, raw['xyz'][0], atol=1.e-6, rtol=0)
        torch.testing.assert_close(covariance, torch.eye(3), atol=1.e-5, rtol=0)
        for one, two in zip(children, second[0]):
            for name in rb.RAW_FIELDS:
                torch.testing.assert_close(one[name], two[name], atol=0, rtol=0)
        self.assertTrue(all(item['kind'] == 'inherited_covariance_prior' for item in domains))

    def test_nested_support_checkpoint_is_recursive_and_has_no_tensor_aliases(self):
        raw = raw_point()
        transaction = rb.BirthTransaction(raw, torch.tensor([0]), torch.tensor([0]), torch.ones(1),
            raw['xyz'], raw['rotation'], raw['logscale'].exp(), torch.tensor([3]),
            ('fit1', 'fit2'), ('validation',), 0, 1, torch.tensor([9]),
            {'nested': [{'evidence': torch.tensor([7.])}]}, [rb.covariance_support(raw, 0)])
        state = transaction.capture()
        restored = rb.BirthTransaction.restore(state, 'cpu')
        restored.support_domains[0]['xyz'][0, 0] = 99
        restored.region_diagnostics['nested'][0]['evidence'][0] = 12
        self.assertNotEqual(float(state['support_domains'][0]['xyz'][0, 0]), 99)
        self.assertEqual(float(transaction.region_diagnostics['nested'][0]['evidence'][0]), 7)
        torch.testing.assert_close(restored.group_ids, torch.tensor([9]))

    def test_coarse_model_and_deep_optimizer_state_move_before_restore(self):
        raw = raw_point()
        transaction = rb.BirthTransaction(raw, torch.tensor([0]), torch.tensor([0]), torch.ones(1),
            raw['xyz'], raw['rotation'], raw['logscale'].exp(), torch.tensor([3]),
            ('fit1', 'fit2'), ('validation',), 0, 1, torch.tensor([9]), {}, [])
        snapshot = dict(transaction=transaction.capture(), model=(torch.ones(1, 3),
            {'optimizer': [{'exp_avg': torch.ones(1, 3)}]}))
        # Meta provides a distinct device in a CPU-only test, catching code that
        # forwards an unchanged CPU capture to libgs.restore (which never moves
        # model_args itself).
        model = NS(restore=Mock())
        module = NS(device=torch.device('meta'), config=NS(sh_degree=1, gaussian_stage2=object()))
        with patch.object(MODULE, 'BirthGaussianModel', return_value=model):
            MODULE.Module._restore_birth_pool(module, snapshot)
        moved = model.restore.call_args.args[0]
        self.assertEqual(moved[0].device.type, 'meta')
        self.assertEqual(moved[1]['optimizer'][0]['exp_avg'].device.type, 'meta')
        self.assertEqual(module.birth_transaction.sources.device.type, 'meta')
        self.assertEqual(module.births.source_rows.device.type, 'meta')
        self.assertEqual(snapshot['model'][0].device.type, 'cpu')

    def test_next_cohort_extends_deadline_by_a_full_training_budget(self):
        raw = raw_point()
        transaction = NS(fit_names=('fit',), capture=lambda: {'nested': raw})
        module = NS(birth_cohort=0, birth_cohort_start=200, birth_end_step=800,
            trainer=NS(global_step=800), birth_control={}, birth_transaction=transaction,
            config=NS(birth=NS(residual=rb.ResidualBirthConfig(),), rigid_tree=NS(birth_steps=600)),
            transaction_camera_views=lambda: {'fit': object()},
            births=NS(_xyz=torch.zeros(1, 3), capture=lambda: {'optimizer': {'step': torch.tensor(600)}}),
            coarse_birth_snapshot=[])
        next_transaction = object()
        module._install_birth_transaction = Mock()
        with patch.object(MODULE, 'fitting_errors', return_value=torch.tensor([1.])), \
                patch.object(MODULE, 'refine_transaction', return_value=(next_transaction, {'split_parents': 1})):
            advanced = MODULE.Module._advance_birth_cohort(module)
        self.assertTrue(advanced)
        self.assertEqual(module.birth_cohort, 1)
        self.assertEqual(module.birth_cohort_start, 800)
        self.assertEqual(module.birth_end_step, 1400)
        self.assertEqual(len(module.coarse_birth_snapshot), 1)
        module._install_birth_transaction.assert_called_once_with(next_transaction)
        # The copied optimizer belongs to the coarse rollback snapshot only.
        self.assertEqual(int(module.coarse_birth_snapshot[0]['model']['optimizer']['step']), 600)

    def _pool_selection(self, fine_errors):
        coarse = NS(_xyz=torch.zeros(1, 3), prune_points=Mock())
        fine = NS(_xyz=torch.zeros(2, 3), prune_points=Mock())
        transaction = NS(fit_names=('fit',), validation_names=('validation',))
        committed = []
        module = NS(device=torch.device('cpu'), births=fine, birth_transaction=transaction,
            coarse_birth_snapshot=[{'cohort': 0}], birth_cohort=1, birth_control={},
            gaussians=NS(_xyz=torch.zeros(4, 3)), motion=NS(record_final_diagnostics=Mock()),
            config=NS(birth=NS(residual=rb.ResidualBirthConfig())),
            transaction_camera_views=lambda: {'fit': object(), 'validation': object()})
        def restore(snapshot):
            module.births = coarse; module.birth_transaction = transaction
        module._restore_birth_pool = restore
        module._commit_opacity_and_retire = lambda **kwargs: committed.append(module.births)
        def admit(module, *args, **kwargs):
            self.assertEqual(committed, [])
            errors = [.9, .9] if module.births is coarse else fine_errors
            return (rb.CommitTrial(torch.arange(4), torch.tensor([0]), torch.tensor([0]), {}, 100, 50, 50, 0),
                dict(validation_committed_mse=errors, accepted=1, full_pool_fitting_mse=errors,
                     full_pool_validation_mse=errors, full_pool_baseline_fitting_mse=[1., 1.]))
        with patch.object(MODULE, 'fitting_errors', return_value=torch.tensor([1.])), \
                patch.object(MODULE, 'admit_residual_births', admit):
            MODULE.Module._select_residual_births(module)
        return module, coarse, fine, committed

    def test_harmful_fine_pool_rolls_back_before_any_old_model_mutation(self):
        module, coarse, fine, committed = self._pool_selection([.8, 1.1])
        self.assertIs(module.births, coarse)
        self.assertEqual(committed, [])
        self.assertEqual(module.birth_control['selected_cohort'], 0)
        fine.prune_points.assert_not_called()

    def test_legal_fine_pool_can_win_with_a_normal_camera_tradeoff(self):
        module, coarse, fine, committed = self._pool_selection([.8, .98])
        self.assertIs(module.births, fine)
        self.assertEqual(committed, [])
        self.assertEqual(module.birth_control['selected_cohort'], 1)
        coarse.prune_points.assert_not_called()


# ----- test_child_dc_initializer -----
class ChildDCMathTests(unittest.TestCase):
    def test_explicit_native_moments_match_double_matmul_with_overlap_and_signed_residual(self):
        generator=torch.Generator().manual_seed(813)
        flat=torch.rand(3,2,10007,dtype=torch.float64,generator=generator)
        # Opposing residuals exercise cancellation in g; siblings overlap.
        residual=torch.randn(10007,dtype=torch.float64,generator=generator)
        h,b=_pair_image_moments(flat,residual)
        torch.testing.assert_close(h,flat@flat.transpose(1,2),rtol=2e-13,atol=1e-12)
        torch.testing.assert_close(b,flat@residual,rtol=2e-13,atol=1e-12)
        self.assertTrue(bool((h[:,0,1]>0).all()))
        torch.testing.assert_close(h[:,0,1],h[:,1,0],rtol=0,atol=0)

    def test_true_sibling_cross_term_recovers_both_colours(self):
        j=torch.tensor([[1.,.9],[.2,.7],[.7,.1]],dtype=torch.float64)
        truth=torch.tensor([.3,-.2],dtype=torch.float64);g=j.T@j;b=j.T@(j@truth)
        got,_=solve_box2(g,b,torch.tensor([-1.,-1.]),torch.tensor([1.,1.]))
        torch.testing.assert_close(got,truth,atol=1e-12,rtol=0)
        self.assertGreater(float((b/g.diag()-truth).abs().max()),.1)

    def test_bounded_edge_is_resolved_with_coupled_free_coordinate(self):
        g=torch.tensor([[2.,1.],[1.,2.]],dtype=torch.float64);b=g@torch.tensor([2.,-.1],dtype=torch.float64)
        got,_=solve_box2(g,b,torch.tensor([-.5,-.5]),torch.tensor([.5,.5]))
        torch.testing.assert_close(got,torch.tensor([.5,.5],dtype=torch.float64),rtol=0,atol=1e-12)

    def test_singular_duplicate_responses_and_zero_response_do_not_explode(self):
        g=torch.stack((torch.ones(2,2),torch.zeros(2,2))).double();b=torch.tensor([[.4,.4],[0.,0.]],dtype=torch.float64)
        got,r=solve_box2(g,b,-torch.ones(2,2),torch.ones(2,2))
        self.assertAlmostEqual(float(got[0].sum()),.4,places=12)
        self.assertEqual(got[1].tolist(),[0.,0.]);self.assertEqual(r['zero_response_blocks'],1)

    def test_box_optima_have_independent_KKT_certificate(self):
        gen=torch.Generator().manual_seed(73)
        for _ in range(30):
            j=torch.randn(5,2,dtype=torch.float64,generator=gen);y=torch.randn(5,dtype=torch.float64,generator=gen)
            g,b=j.T@j,j.T@y;lo=torch.tensor([-.3,-.2]);hi=torch.tensor([.4,.7])
            w,_=solve_box2(g,b,lo,hi);gradient=g@w-b
            self.assertTrue(bool((w>=lo.double()).all()&(w<=hi.double()).all()))
            free=(w>lo+1e-7)&(w<hi-1e-7)
            self.assertTrue(bool((gradient[free].abs()<1e-9).all()))
            self.assertTrue(bool((gradient[w<=lo+1e-7]>=-1e-9).all()))
            self.assertTrue(bool((gradient[w>=hi-1e-7]<=1e-9).all()))

    def test_zero_must_be_feasible_and_indefinite_gram_is_rejected(self):
        with self.assertRaises(ValueError):solve_box2(torch.eye(2),torch.zeros(2),torch.ones(2),torch.ones(2)*2)
        with self.assertRaises(ValueError):solve_box2(torch.diag(torch.tensor([-1.,1.])),torch.zeros(2),-torch.ones(2),torch.ones(2))


class ChildDCIntegrationTests(unittest.TestCase):
    def fixture(self):
        import importlib
        from test_reincarnation import source_controller_fixture as fixture
        from pipeline.model.birth_radiance import _response_scene
        from pipeline.model.birth_quantization import model_raw
        m,views=fixture();M=importlib.import_module('pipeline.module')
        M.Module._advance_dual_stage(m,800);m._activate_dual_branch('A')
        class Renderer:
            collect_contributions=False
            last_contributions=None
            def __init__(self,old):self.gaussians=old
            def __call__(self,view,bg,override_color=None):
                g=self.gaussians;x=torch.linspace(-.2,.7,16)[None,None,:];y=torch.linspace(-.2,.7,16)[None,:,None]
                scales=g.get_scaling;xyz=g.get_xyz
                z=((x-xyz[:,0,None,None])/scales[:,0,None,None]).square()+((y-xyz[:,1,None,None])/scales[:,1,None,None]).square()
                alpha=g.get_opacity[:,0,None,None]*torch.exp(-.5*z)
                trans=torch.cumprod(torch.cat((torch.ones_like(alpha[:1]),1-alpha),0),0)[:-1]
                colours=(.5+.28209479177387814*g.get_features[:,0]).clamp_min(0) if override_color is None else override_color
                return {'render':torch.einsum('nhw,nc->chw',alpha*trans,colours)}
        m.renderer=Renderer(m.gaussians)
        def forward(view):
            previous=m.renderer.gaussians
            try:
                m.renderer.gaussians=_response_scene(m)[0]
                return m.renderer(view,m.background)
            finally:m.renderer.gaussians=previous
        m.forward=forward
        leaf=m.births._features_dc;initial=leaf.detach().clone();ids=m.births.detail_rows
        with torch.no_grad():
            leaf[ids[0]]+=.3;leaf[ids[1]]-=.2;m.births._clear_reincarnation_cache()
            for view in views.values():view.image=m.forward(view)['render'].clone()
            leaf.copy_(initial);m.births._clear_reincarnation_cache()
        return m,views,model_raw

    def test_single_proposal_improves_and_keeps_mother_geometry_optimizer_untouched(self):
        m,views,raw=self.fixture();before=raw(m.births);leaf=m.births._features_dc
        optimizer=m.births.optimizer
        optimizer.state[leaf]=dict(step=torch.tensor(3.),exp_avg=torch.full_like(leaf,.17),exp_avg_sq=torch.full_like(leaf,.03))
        saved={k:v.clone() for k,v in optimizer.state[leaf].items()};leaf.grad=torch.full_like(leaf,.37);saved_grad=leaf.grad.clone()
        with patch.object(optimizer,'step',side_effect=AssertionError('hidden optimizer step')):
            report=initialize_child_dc_once(m,views)
        self.assertTrue(report['accepted']);self.assertEqual(report['SGD_steps'],0)
        self.assertIs(leaf,m.births._features_dc)
        for key,value in saved.items():torch.testing.assert_close(optimizer.state[leaf][key],value,rtol=0,atol=0)
        torch.testing.assert_close(leaf.grad,saved_grad,rtol=0,atol=0)
        after=raw(m.births)
        for name in ('xyz','rotation','logscale','opacity','sh_rest'):torch.testing.assert_close(before[name],after[name],rtol=0,atol=0)
        torch.testing.assert_close(before['sh_dc'][m.births.reincarnation_rows],after['sh_dc'][m.births.reincarnation_rows],rtol=0,atol=0)
        self.assertLess(report['after_mean_mse'],report['before_mean_mse'])

    def test_real_codec_callback_rejection_rolls_back_once_without_search(self):
        m,views,_=self.fixture();initial=m.births._features_dc.detach().clone();calls=[]
        def validate():
            calls.append(True);value=1. if len(calls)==1 else 2.
            return dict(mse=value,per_view_mse=[value],codec_bytes=123)
        report=initialize_child_dc_once(m,views,validate_scene=validate)
        self.assertFalse(report['accepted']);self.assertEqual(len(calls),2)
        self.assertEqual(report['after_validation']['codec_bytes'],123)
        torch.testing.assert_close(initial,m.births._features_dc,rtol=0,atol=0)


# ----- test_population_refit -----
def model(xyz):
    result = BirthGaussianModel(1)
    count = len(xyz)
    values = (torch.tensor(xyz, dtype=torch.float32), torch.tensor([[1., 0., 0., 0.]]).repeat(count, 1),
              torch.zeros(count, 3), torch.zeros(count, 1), torch.zeros(count, 1, 3), torch.zeros(count, 3, 3))
    for attribute, value in zip(RAW_ATTRIBUTES, values):
        setattr(result, attribute, value)
    return result


class PopulationRefitTests(unittest.TestCase):
    def test_real_codec_selected_and_fallback_replay_non_grid_quaternion_exactly(self):
        # This concrete pair changed q by >3e-4 when decoded normalized q was
        # reused as the next quantizer input. Preserve its continuous preimage.
        xyz = torch.cartesian_prod(torch.tensor([.1, .3, .6, .9]),
            torch.tensor([.1, .9]), torch.tensor([.1, .9])).repeat_interleave(2, 0)
        old = model(xyz.tolist())
        old._rotation[:] = torch.tensor([-.28191742300987244, .8100713491439819,
                                        -.48882603645324707, .1592356562614441])
        births = model([[.43, .52, .61]])
        births._rotation[:] = torch.tensor([.20900920033454895, .11839142441749573,
                                           -.4188501536846161, -1.0293842554092407])
        births._scaling.fill_(-2.3456); births._opacity.fill_(-.76543)
        births._features_dc.fill_(.12345); births._features_rest.fill_(-.24567)
        raw = model_raw(births)
        transaction = rb.BirthTransaction(raw, torch.tensor([0]), torch.tensor([-1]), torch.ones(1),
            raw['xyz'], raw['rotation'], raw['logscale'].exp(), torch.tensor([3]),
            ('fit1', 'fit2'), ('fit1', 'fit2'), 1, 0, torch.tensor([8]), {}, [])
        motion = SpatialGatedMotion(RigidTreeConfig(motion_steps=200, birth_steps=600,
            rate_loss_weight=0., relative_motion_loss_weight=0.))
        motion.setup(old._xyz, old._rotation, 1., frame_index=1)
        module = NS(device=torch.device('cpu'), current_frame=1, gaussians=old, births=births,
            motion=motion, opacity_update=None, birth_transaction=transaction,
            config=NS(sh_degree=1, birth=MODULE.BirthConfig(rotation_updates=True)),
            datamodule=NS(cameras_extent=1.))
        trial = rb.encode_trial(module, torch.tensor([True]), torch.zeros(len(xyz), dtype=torch.bool))
        selected = rb.selected_population(module, trial)
        torch.testing.assert_close(selected.raw['rotation'], raw['rotation'], atol=0, rtol=0)
        torch.testing.assert_close(selected.seed_rotation, trial.raw['rotation'][len(xyz):], atol=0, rtol=0)
        module.birth_transaction = selected
        snapshot = selected.capture()
        def replay(working):
            for name, attribute in zip(RAW_FIELDS, RAW_ATTRIBUTES):
                setattr(module.births, attribute, working.raw[name].clone())
            module.births.source_rows = working.sources.clone()
            module.birth_transaction = working
            view = MODULE.Module.make_birth_view(module)
            return materialize_quantized_gaussians(working.fixed_reference, view)[0]
        # Demonstrate that the old decoded-q initialization really fails on
        # this example, rather than testing only a benign grid-aligned q.
        bad = rb.BirthTransaction.restore(snapshot, 'cpu')
        bad.raw['rotation'] = trial.raw['rotation'][len(xyz):].clone()
        bad_scene = replay(bad)
        self.assertGreater(float((bad_scene['rotation']-trial.raw['rotation']).abs().max()), 1.e-5)
        for working in (selected, rb.BirthTransaction.restore(snapshot, 'cpu')):
            scene = replay(working)
            for name in RAW_FIELDS:
                torch.testing.assert_close(scene[name], trial.raw[name], atol=0, rtol=0)

    def fixture(self):
        old = model([[0., 0., 3.], [10., 0., 3.], [20., 0., 3.]])
        births = model([[19.9, 0., 3.]])
        raw = model_raw(births)
        transaction = rb.BirthTransaction(raw, torch.tensor([1]), torch.tensor([1]), torch.ones(1),
            raw['xyz'], raw['rotation'], raw['logscale'].exp(), torch.tensor([3]),
            ('fit1', 'fit2'), ('fit1', 'fit2'), 0, 1, torch.tensor([8]), {}, [])
        module = NS(device=torch.device('cpu'), gaussians=old, births=births,
            birth_transaction=transaction, config=NS(sh_degree=1, birth=MODULE.BirthConfig(rotation_updates=True),
                effective_motion_observations=False), datamodule=NS(cameras_extent=1.))
        keep = torch.tensor([0, 2])
        combined = {name: torch.cat((model_raw(old)[name][keep], value)) for name, value in raw.items()}
        trial = rb.CommitTrial(keep, torch.tensor([0]), torch.tensor([1]), combined, 3000, 1000, 1000, 0)
        return module, trial

    def test_all_twenty_standard_training_cameras_fit_and_select(self):
        class Dataset:
            items = [NS(path=f'cam{i:02}.png') for i in range(1, 21)]
            def __getitem__(self, index):
                return NS(to=lambda device: self.items[index])
        dataset = Dataset()
        module = NS(_transaction_views=None, birth_transaction=None, device=torch.device('cpu'),
            datamodule=NS(get_train_dataset=lambda: dataset,
                          get_test_dataset=lambda: NS(items=[NS(path='cam00.png')])))
        views = MODULE.Module.transaction_camera_views(module)
        self.assertEqual(len(views), 20)
        self.assertEqual(module._transaction_fit_names, module._transaction_validation_names)
        self.assertEqual(len(module._transaction_fit_names), 20)
        self.assertNotIn('cam00.png', views)

    def test_mean_improvement_allows_normal_per_camera_tradeoffs(self):
        self.assertTrue(rb.improves(torch.tensor([.7, 1.1]), torch.ones(2), 1.e-4))
        self.assertFalse(rb.improves(torch.tensor([.9, 1.2]), torch.ones(2), 1.e-4))

    def test_selected_structure_hides_old_rows_without_mutating_and_fixes_predictor(self):
        module, trial = self.fixture()
        original = model_raw(module.gaussians)
        module.birth_transaction = rb.selected_population(module, trial)
        module.births.source_rows = module.birth_transaction.sources.clone()
        # Move the candidate next to the other survivor: terminal quantization
        # must retain its selected source, not perform nearest-source reselection.
        module.births._xyz[0, 0] = .1
        view = MODULE.Module.make_birth_view(module)
        torch.testing.assert_close(view.predictor['xyz'], torch.tensor([[20., 0., 3.]]))
        preview = rb.encode_trial(module, torch.tensor([True]),
                                 module.birth_transaction.provisional_deleted, encode=False)
        self.assertEqual(preview.sources.tolist(), [1])
        self.assertEqual(preview.keep.tolist(), [0, 2])
        class Renderer:
            gaussians = module.gaussians
            def __call__(self, viewpoint, background):
                self.seen = self.gaussians.get_xyz.clone()
                return {'render': torch.zeros(3, 2, 2)}
        module.renderer = Renderer()
        module.background = torch.zeros(3)
        module.frame_prepared = module.motion_committed = True
        module.frame_finalized = False
        module.opacity_update = None
        module.make_birth_view = lambda: MODULE.Module.make_birth_view(module)
        MODULE.Module.forward(module, object())
        torch.testing.assert_close(module.renderer.seen[:2], original['xyz'][trial.keep])
        self.assertEqual(len(module.gaussians._xyz), 3)
        for name, value in model_raw(module.gaussians).items():
            torch.testing.assert_close(value, original[name], atol=0, rtol=0)

    def test_terminal_checkpoint_preserves_structure_and_validated_fallback(self):
        module, trial = self.fixture()
        transaction = rb.selected_population(module, trial)
        state = dict(active=True, selected=dict(transaction=transaction.capture(), trial=trial.capture(),
                    budget=5000, selected_mse=[.8, .8]))
        restored = rb.BirthTransaction.tensor_tree(state, 'cpu')
        tx = rb.BirthTransaction.restore(restored['selected']['transaction'], 'cpu')
        validated = rb.CommitTrial.restore(restored['selected']['trial'], 'cpu')
        self.assertEqual(tx.provisional_deleted.tolist(), [False, True, False])
        self.assertEqual(tx.fixed_keep.tolist(), [0, 2])
        self.assertEqual(tx.sources.tolist(), [1])
        for name in RAW_FIELDS:
            torch.testing.assert_close(tx.fixed_reference[name], validated.raw[name][:2])
        tx.fixed_reference['xyz'][0, 0] = 99
        self.assertEqual(float(state['selected']['transaction']['fixed_reference']['xyz'][0, 0]), 0)

    def test_bad_final_refit_restores_validated_pool_before_physical_commit(self):
        module, trial = self.fixture()
        transaction = rb.selected_population(module, trial)
        module.birth_transaction = transaction
        module.births.source_rows = transaction.sources.clone()
        module.selected_birth_state = dict(transaction=transaction.capture(), trial=trial.capture(),
            budget=5000, baseline_mse=[1., 1.], selected_mse=[.8, .8])
        module.birth_control = {'population_refit': {}}
        module.birth_cohort = 2; module.birth_cohort_start = 1400
        module.trainer = NS(global_step=2000)
        module.transaction_camera_views = lambda: {'fit1': object(), 'fit2': object()}
        installed, committed = [], []
        module._install_birth_transaction = lambda transaction: installed.append(transaction)
        module._commit_population_trial = lambda chosen: committed.append(chosen)
        with patch.object(MODULE, 'encode_birth_trial', return_value=trial), \
                patch.object(MODULE, 'birth_validation_errors', return_value=torch.tensor([.9, .9])), \
                patch.object(MODULE, 'fitting_errors', return_value=torch.tensor([.8, .8])):
            MODULE.Module._finish_population_refit(module)
        self.assertEqual(len(installed), 1)
        self.assertEqual(len(committed), 1)
        self.assertFalse(module.birth_control['population_refit']['accepted_refit'])
        torch.testing.assert_close(installed[0].raw['xyz'], trial.raw['xyz'][2:])
        self.assertEqual(len(module.gaussians._xyz), 3)
        self.assertEqual(committed[0].keep.tolist(), [0, 2])


if __name__ == '__main__':
    unittest.main()
