"""Reincarnation tests."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import copy
import importlib
import math
import tempfile
import unittest
from types import MethodType, SimpleNamespace, SimpleNamespace as NS
from unittest.mock import Mock, patch

import torch
from torch import nn

from libgs.utils.sh import eval_sh
from pipeline.early_replacement import EarlyReplacementController
from pipeline.model import residual_birth as rb
from pipeline.model.birth_quantization import (
    LiveReplacementView,
    QuantizedNewGaussianView,
    RAW_ATTRIBUTES,
    RAW_FIELDS,
    materialize_quantized_gaussians,
    model_raw,
)
from pipeline.model.early_replacement_model import EarlyReplacementModel
from pipeline.model.gaussian import BirthGaussianModel, GaussianModel
from pipeline.model.persistent_newborn_model import PersistentNewbornModel
from pipeline.model.predictive_reincarnation_model import PredictiveReincarnationModel
from pipeline.model.quick_source_residual_model import QuickSourceResidualModel
from pipeline.model.reincarnation_model import (
    C0,
    C1,
    ReincarnationModel,
    _empty_raw,
    project_sh1_sphere,
    sh1_sphere_range,
)
from pipeline.model.rigid_tree_birth_codec import (
    ATTRIBUTES,
    decode_birth_attributes,
    encode_birth_attributes,
    encode_birth_symbols,
)
from pipeline.model.shadow_birth_initializer import (
    density_coefficients,
    solve_density_quartic,
)
from pipeline.model.shadow_source_residual_model import ShadowSourceResidualModel
from pipeline.model.source_residual_model import SourceResidualModel
from pipeline.model.support_reincarnation_proposal import (
    camera_complementary_groups,
    propose_support_reincarnations,
    score_support_responsibility,
)
from test_surface import fixture as base_fixture

# ----- test_reincarnation_model -----
def arguments():
    return NS(percent_dense=.01, position_lr_init=.001, feature_lr=.01,
              opacity_lr=.02, scaling_lr=.005, rotation_lr=.001)


def config():
    return NS(new_xyz_step_extent=.001, new_rotation_step=.003,
        new_logscale_step=.02, new_opacity_step=.02, new_sh_step=.006, rotation_updates=True)


def reincarnation_fixture(dtype=torch.float64, novel_count=2):
    anchor = _empty_raw(2, 1, 'cpu', dtype)
    anchor['xyz'][:] = torch.tensor([[.1, -.1, 3.], [-.2, .2, 3.3]], dtype=dtype)
    anchor['logscale'][:] = torch.tensor([[.2, .1, .01], [.15, .09, .008]], dtype=dtype).log()
    anchor['opacity'][:] = torch.tensor([[.2837], [1.073]], dtype=dtype)
    anchor['rotation'][1] = torch.tensor([.9, .1, -.2, .05], dtype=dtype)
    anchor['rotation'][1] /= anchor['rotation'][1].norm()
    anchor['sh_dc'][:] = torch.tensor([[[-0., .2, -.3]], [[.1, -.2, .3]]], dtype=dtype)
    anchor['sh_rest'][:] = torch.tensor([[[.03, -.02, .01], [.04, .02, -.03], [-.02, .01, .04]]], dtype=dtype)
    specs = [dict(old_row=index, raw={name: value[index].clone() for name, value in anchor.items()},
                  tangent_offset=torch.tensor([.04, .01], dtype=dtype), responsibility=1.) for index in range(2)]
    novel = _empty_raw(novel_count, 1, 'cpu', dtype)
    novel['xyz'][:, 2] = 3.1
    novel['opacity'].fill_(-2.)
    novel['logscale'].fill_(-3.)
    model = ReincarnationModel(1).install_reincarnations(specs, novel_raw=novel)
    model.spatial_lr_scale = 1.
    return model, anchor


def bytes_equal(one, two):
    return one.detach().cpu().contiguous().numpy().tobytes() == two.detach().cpu().contiguous().numpy().tobytes()


def weighted_loss(model):
    result = model.reincarnation_phenotype()
    weight = torch.arange(1, len(model._xyz)+1, dtype=model._xyz.dtype)[:, None]
    loss = model._xyz.new_zeros(())
    for value in result.values():
        w = weight.reshape(len(weight), *([1]*(value.ndim-1)))
        loss = loss+(value*w).sum()
    return loss


class SphereTests(unittest.TestCase):
    def test_sh1_vector_signs_and_full_sphere_extrema_match_eval_sh(self):
        dc = torch.tensor([[[.4, -.3, .2]]], dtype=torch.float64)
        rest = torch.tensor([[[.2, -.1, .3], [.1, .4, -.2], [-.3, .2, .1]]], dtype=torch.float64)
        low, high = sh1_sphere_range(dc, rest)
        b = C1*torch.stack((-rest[:, 2], -rest[:, 0], rest[:, 1]), -1)
        features = torch.cat((dc, rest), 1).transpose(1, 2)
        for channel in range(3):
            direction = b[0, channel]/b[0, channel].norm()
            positive = eval_sh(1, features, direction[None])+.5
            negative = eval_sh(1, features, -direction[None])+.5
            torch.testing.assert_close(positive[0, channel], high[0, channel])
            torch.testing.assert_close(negative[0, channel], low[0, channel])

    def test_triangle_projection_and_joint_detail_constraint_are_not_rgb_clips(self):
        dc = torch.full((1, 1, 3), 5., dtype=torch.float64)
        rest = torch.full((1, 3, 3), 4., dtype=torch.float64)
        detail = torch.full((1, 3), 3., dtype=torch.float64)
        bounded_dc, bounded_rest, bounded_detail = project_sh1_sphere(dc, rest,
            torch.zeros(1, 3, dtype=torch.float64), torch.ones(1, 3, dtype=torch.float64), detail)
        low, high = sh1_sphere_range(bounded_dc, bounded_rest, bounded_detail)
        self.assertTrue(bool((low >= -1.e-14).all() & (high <= 1+1.e-14).all()))
        self.assertTrue(bool(torch.isfinite(bounded_rest).all()))
        self.assertTrue(bool((bounded_rest.abs() < rest.abs()).all()))

    def test_physical_prior_keeps_real_hdr_and_never_uses_codec_source(self):
        model, anchor = reincarnation_fixture(novel_count=0)
        model.support_anchor_raw['sh_dc'][0].fill_(4.)
        model.p_common_dc.data[0].fill_(4.)
        prior_low, prior_high = sh1_sphere_range(model.support_anchor_raw['sh_dc'], model.support_anchor_raw['sh_rest'])
        self.assertGreater(float(prior_high[0].max()), 1.)
        model.p_common_dc.data.add_(100.)
        model.p_common_sh1.data.add_(20.)
        model.project_parameters()
        low, high = sh1_sphere_range(model.p_common_dc, model.p_common_sh1)
        self.assertTrue(bool((low >= prior_low.clamp_max(0)-1.e-12).all()))
        self.assertTrue(bool((high <= prior_high.clamp_min(1)+1.e-12).all()))

    def test_sh0_works_but_higher_degree_is_explicitly_unsupported(self):
        raw = _empty_raw(1, 0, 'cpu', torch.float64)
        raw['xyz'][0, 2] = 3.
        model = ReincarnationModel(0).install_reincarnations([dict(old_row=0, raw={k:v[0] for k,v in raw.items()})])
        self.assertEqual(model.birth_parameter_value('sh_rest').shape, (1, 0, 3))
        with self.assertRaises(NotImplementedError):
            ReincarnationModel(3)


class NativeSupportTests(unittest.TestCase):
    def test_coarse_six_native_fields_are_bitwise_equal_before_and_after_noop_projection(self):
        model, anchor = reincarnation_fixture()
        arbitrary = {name: value.clone() for name, value in anchor.items()}
        arbitrary['sh_dc'].fill_(150.)
        for project in (False, True):
            if project: model.project_parameters()
            phenotype = model_raw(model)
            quantized = model.reincarnation_quantized_raw(config(), 1., arbitrary,
                torch.zeros(len(model._xyz), dtype=torch.long))
            for name in RAW_FIELDS:
                self.assertTrue(bytes_equal(phenotype[name][2:], anchor[name]), name)
                self.assertTrue(bytes_equal(quantized[name][2:], anchor[name]), name)

    def test_common_material_learns_while_p_geometry_alpha_and_raw_slots_do_not(self):
        model, anchor = reincarnation_fixture()
        model.training_setup(arguments())
        weighted_loss(model).backward()
        for parameter in (model.p_common_dc, model.p_common_sh1):
            self.assertGreater(float(parameter.grad.abs().sum()), 0.)
        for attr in RAW_ATTRIBUTES:
            parameter = getattr(model, attr)
            if parameter.grad is not None:
                torch.testing.assert_close(parameter.grad[2:], torch.zeros_like(parameter.grad[2:]), rtol=0, atol=0)
        model.optimizer.step(); model.project_parameters()
        after = model_raw(model)
        for name in ('xyz', 'rotation', 'logscale', 'opacity'):
            self.assertTrue(bytes_equal(after[name][2:], anchor[name]), name)
        self.assertFalse(bytes_equal(after['sh_dc'][2:], anchor['sh_dc']))
        rates = {group['name']: group['lr'] for group in model.optimizer.param_groups}
        self.assertEqual(rates['p_common_dc'], arguments().feature_lr)
        self.assertEqual(rates['p_common_sh1'], arguments().feature_lr/20)

    def test_per_view_novel_row_detach_does_not_detach_common_p(self):
        model, _ = reincarnation_fixture()
        raw = model.reincarnation_phenotype()
        active = torch.tensor([True, False, True, True])
        objective = model._xyz.new_zeros(())
        for value in raw.values():
            shape = (len(value),)+tuple(1 for _ in range(value.ndim-1))
            masked = torch.where(active.reshape(shape), value, value.detach())
            objective = objective+masked.sum()
        objective.backward()
        for attr in RAW_ATTRIBUTES:
            gradient = getattr(model, attr).grad
            if gradient is not None:
                torch.testing.assert_close(gradient[1], torch.zeros_like(gradient[1]), rtol=0, atol=0)
        self.assertGreater(float(model.p_common_dc.grad.abs().sum()), 0.)
        self.assertGreater(float(model.p_common_sh1.grad.abs().sum()), 0.)

    def test_novel_quantized_material_remains_sphere_ldr_with_hdr_predictor(self):
        model, anchor = reincarnation_fixture()
        model._features_dc.data[:2].fill_(100.)
        model._features_rest.data[:2].fill_(20.)
        anchor['sh_dc'].fill_(500.)
        anchor['sh_rest'].fill_(-30.)
        raw = model.reincarnation_quantized_raw(config(), 1., anchor, torch.zeros(4, dtype=torch.long))
        low, high = sh1_sphere_range(raw['sh_dc'][:2], raw['sh_rest'][:2])
        self.assertTrue(bool((low >= -1.e-12).all() & (high <= 1+1.e-12).all()))


class FineAndLifecycleTests(unittest.TestCase):
    def fine(self):
        model, _ = reincarnation_fixture()
        model.p_common_dc.data.add_(.05)
        return model.install_families_from_coarse([0], tangent_offsets=torch.tensor([[.04, .01]], dtype=torch.float64))

    def test_fine_uses_original_support_and_exact_requested_half_displacement(self):
        coarse, anchor = reincarnation_fixture()
        coarse.p_common_dc.data.add_(.1)
        # A previously fitted free/raw P geometry must not become the anchor.
        coarse._xyz.data[2:] += 20.
        fine = coarse.install_families_from_coarse([0], tangent_offsets=torch.tensor([[.04, .01]], dtype=torch.float64))
        raw = model_raw(fine); rows = fine.family_rows[0]
        torch.testing.assert_close(raw['xyz'][rows[0]], anchor['xyz'][0], rtol=0, atol=0)
        torch.testing.assert_close(raw['xyz'][rows[1]]-raw['xyz'][rows[0]], torch.tensor([.04, .01, 0.], dtype=torch.float64), rtol=0, atol=1.e-15)
        torch.testing.assert_close((raw['sh_dc'][rows[1]]+raw['sh_dc'][rows[2]])*.5, raw['sh_dc'][rows[0]], rtol=0, atol=1.e-15)
        torch.testing.assert_close(fine.support_anchor_raw['sh_dc'], anchor['sh_dc'], rtol=0, atol=0)
        self.assertFalse(fine.family_gate_logit.requires_grad)
        with self.assertRaisesRegex(ValueError, 'ellipse'):
            coarse.install_families_from_coarse([0], tangent_offsets=torch.tensor([[.3, .2]], dtype=torch.float64))

    def test_joint_common_and_detail_stay_inside_sphere_and_do_not_reset_prior(self):
        model = self.fine()
        prior = model.support_anchor_raw['sh_dc'].clone()
        model.p_common_dc.data.add_(20.)
        model.p_common_sh1.data.add_(8.)
        model.family_contrast.data.fill_(50.)
        model.project_parameters()
        raw = model_raw(model)
        lo0, hi0 = sh1_sphere_range(prior, model.support_anchor_raw['sh_rest'])
        for root, rows in enumerate(model.reincarnation_groups):
            low, high = sh1_sphere_range(raw['sh_dc'][rows], raw['sh_rest'][rows])
            self.assertTrue(bool((low >= lo0[root].clamp_max(0)-1.e-12).all()))
            self.assertTrue(bool((high <= hi0[root].clamp_min(1)+1.e-12).all()))
        self.assertTrue(bytes_equal(model.support_anchor_raw['sh_dc'], prior))

    def test_independent_novel_initial_state_survives_mature_a_and_fine_b_reset(self):
        model, _ = reincarnation_fixture()
        original = {name:value.clone() for name,value in model.novel_initial_raw.items()}
        for attr in RAW_ATTRIBUTES:
            getattr(model, attr).data[:2].add_(.1)
        fine = model.install_families_from_coarse([0])
        for name, attr in zip(RAW_FIELDS, RAW_ATTRIBUTES):
            self.assertTrue(bytes_equal(fine.novel_initial_raw[name], original[name]), name)
            getattr(fine, attr).data[:2].copy_(fine.novel_initial_raw[name])
            self.assertTrue(bytes_equal(getattr(fine, attr)[:2], original[name]), name)

    def test_capture_restore_atomic_subset_and_shared_adam_are_independent(self):
        model = self.fine().set_training_stage(True)
        model.training_setup(arguments())
        weighted_loss(model).backward(); model.optimizer.step(); model.optimizer.zero_grad(set_to_none=True); model.project_parameters()
        packet = model.capture()
        restored = ReincarnationModel(1).restore(packet, arguments())
        for name in RAW_FIELDS:
            self.assertTrue(bytes_equal(model_raw(restored)[name], model_raw(model)[name]), name)
        self.assertTrue(restored.family_gate_logit.requires_grad)
        with self.assertRaisesRegex(ValueError, 'atomically'):
            restored.reincarnation_subset(torch.tensor([0, 2, 3]))
        selected = restored.reincarnation_subset(torch.tensor([1, 2, 3, 4]))
        self.assertEqual(selected.novel_count, 1)
        self.assertEqual(selected.family_rows.tolist(), [[1, 2, 3]])
        for name in ('p_common_dc', 'p_common_sh1', 'family_contrast', 'family_tangent_offset', 'family_gate_logit'):
            old, new = getattr(restored, name), getattr(selected, name)
            old_state, new_state = restored.optimizer.state[old], selected.optimizer.state[new]
            torch.testing.assert_close(new_state['exp_avg'], old_state['exp_avg'][:1], rtol=0, atol=0)
        selected.p_common_dc.data.add_(.01)
        self.assertFalse(bytes_equal(selected.p_common_dc, restored.p_common_dc[:1]))
        self.assertTrue(bytes_equal(model.support_anchor_raw['sh_dc'], packet['reincarnation']['anchor_raw']['sh_dc']))


# ----- test_predictive_reincarnation_model -----
def args():
    return NS(percent_dense=.01, position_lr_init=.001, feature_lr=.01,
              opacity_lr=.02, scaling_lr=.005, rotation_lr=.001)


def equal_bytes(a, b):
    return a.detach().cpu().contiguous().numpy().tobytes() == b.detach().cpu().contiguous().numpy().tobytes()


def predictive_fixture(n=2):
    anchor = _empty_raw(2, 1, 'cpu', torch.float64)
    anchor['xyz'][:] = torch.tensor([[.1, -.1, 3.], [-.2, .2, 3.3]], dtype=torch.float64)
    anchor['logscale'][:] = torch.tensor([[.2, .1, .01], [.15, .09, .008]], dtype=torch.float64).log()
    anchor['rotation'][1] = torch.tensor([.9, .1, -.2, .05], dtype=torch.float64)
    anchor['opacity'][:] = torch.tensor([[.2837], [1.073]], dtype=torch.float64)
    anchor['sh_dc'][:] = torch.tensor([[[-0., .2, -.3]], [[.1, -.2, .3]]], dtype=torch.float64)
    anchor['sh_rest'][:] = torch.tensor([[[.03, -.02, .01], [.04, .02, -.03], [-.02, .01, .04]]], dtype=torch.float64)
    novel = _empty_raw(n, 1, 'cpu', torch.float64)
    novel['xyz'][:, 2] = 3.1
    novel['opacity'].fill_(-2.)
    novel['logscale'].fill_(-3.)
    specs = [dict(old_row=i, raw={k:v[i].clone() for k,v in anchor.items()}) for i in range(2)]
    model = PredictiveReincarnationModel(1).install_reincarnations(specs, novel_raw=novel)
    model.spatial_lr_scale = 1.
    return model, anchor


def objective(model):
    raw = model.reincarnation_phenotype()
    result = model._xyz.new_zeros(())
    for value in raw.values():
        result = result+value.sum()
    return result


def grad_is_zero(parameter, rows=None):
    if parameter.grad is None:
        return True
    value = parameter.grad if rows is None else parameter.grad[rows]
    return bool((value == 0).all())


class PredictiveStagesTests(unittest.TestCase):
    def test_phase0_novel_alpha_is_temporary_zero_and_parent_native_six_fields_exact(self):
        model, anchor = predictive_fixture()
        original = {k:v.clone() for k,v in model.novel_initial_raw.items()}
        raw = model_raw(model)
        quantized = model.reincarnation_quantized_raw(config(), 1., anchor,
            torch.zeros(len(model._xyz), dtype=torch.long))
        for value in (raw, quantized):
            self.assertTrue(bool((value['opacity'][:2].sigmoid() == 0).all()))
            for name in RAW_FIELDS:
                self.assertTrue(equal_bytes(value[name][2:], anchor[name]), name)
        objective(model).backward()
        self.assertGreater(float(model.p_common_dc.grad.abs().sum()), 0)
        self.assertGreater(float(model.p_common_sh1.grad.abs().sum()), 0)
        for attr in RAW_ATTRIBUTES:
            self.assertTrue(grad_is_zero(getattr(model, attr)), attr)
        for name, attr in zip(RAW_FIELDS, RAW_ATTRIBUTES):
            self.assertTrue(equal_bytes(getattr(model, attr)[:2], original[name]), name)
        model.set_training_stage(1)
        self.assertTrue(equal_bytes(model_raw(model)['opacity'][:2], original['opacity']))

    def test_details_keep_parent_and_break_symmetry_without_mass_or_displacement(self):
        coarse, _ = predictive_fixture()
        coarse.p_common_dc.data.add_(.05)
        before = model_raw(coarse)
        with patch('pipeline.model.conservative_family.optical_depth_integral', side_effect=AssertionError('mass called')):
            model = coarse.install_details_from_coarse([0])
            raw = model_raw(model)
        rows = model.reincarnation_groups[0]
        self.assertEqual(rows.tolist(), [2, 3, 4])
        for name in RAW_FIELDS:
            self.assertTrue(equal_bytes(raw[name][rows[0]], before[name][2]), name)
        torch.testing.assert_close(raw['xyz'][rows[1:]], raw['xyz'][rows[:1]].expand(2, 3), rtol=0, atol=0)
        torch.testing.assert_close(raw['rotation'][rows[1:]], raw['rotation'][rows[:1]].expand(2, 4), rtol=0, atol=0)
        ratio = (raw['logscale'][rows[1:]]-raw['logscale'][rows[:1]]).exp()
        torch.testing.assert_close(ratio, torch.tensor([[.5, 1., 1.], [1., .5, 1.]], dtype=torch.float64))
        torch.testing.assert_close(raw['opacity'][rows[1:]].sigmoid(), torch.full((2, 1), .1, dtype=torch.float64))
        identity = model.zero_details_raw()
        self.assertTrue(bool((identity['opacity'][rows[1:]].sigmoid() == 0).all()))
        for name in RAW_FIELDS:
            self.assertTrue(equal_bytes(identity[name][rows[0]], before[name][2]))

    def test_phase1_freezes_parent_colour_alpha_and_geometry_but_details_and_novel_learn(self):
        coarse, _ = predictive_fixture()
        model = coarse.install_details_from_coarse([0])
        model.training_setup(args())
        before = model_raw(model)
        objective(model).backward()
        self.assertIsNone(model.p_common_dc.grad)
        self.assertIsNone(model.p_common_sh1.grad)
        for attr in RAW_ATTRIBUTES:
            value = getattr(model, attr)
            self.assertTrue(grad_is_zero(value, model.reincarnation_rows), attr)
            self.assertGreater(float(value.grad[model.detail_rows].abs().sum()), 0., attr)
            self.assertGreater(float(value.grad[:model.novel_count].abs().sum()), 0., attr)
        model.optimizer.step()
        model.project_parameters()
        after = model_raw(model)
        for name in RAW_FIELDS:
            self.assertTrue(equal_bytes(before[name][model.reincarnation_rows], after[name][model.reincarnation_rows]), name)

    def test_phase2_unlocks_parent_alpha_keeps_parent_colour_and_locks_novel_geometry(self):
        coarse, anchor = predictive_fixture()
        model = coarse.install_details_from_coarse([0]).set_training_stage(2)
        raw = model.reincarnation_quantized_raw(config(), 1., anchor, torch.zeros(len(model._xyz), dtype=torch.long))
        sum(value.sum() for value in raw.values()).backward()
        for attr in ('_xyz', '_rotation', '_scaling'):
            parameter = getattr(model, attr)
            self.assertTrue(grad_is_zero(parameter, slice(0, model.novel_count)))
            self.assertTrue(grad_is_zero(parameter, model.reincarnation_rows))
            self.assertGreater(float(parameter.grad[model.detail_rows].abs().sum()), 0.)
        self.assertGreater(float(model._opacity.grad[model.reincarnation_rows].abs().sum()), 0.)
        for attr in ('_opacity', '_features_dc', '_features_rest'):
            self.assertGreater(float(getattr(model, attr).grad[:model.novel_count].abs().sum()), 0.)
        self.assertIsNone(model.p_common_dc.grad)
        self.assertIsNone(model.p_common_sh1.grad)

    def test_quantizer_source_never_changes_parent_or_detail_physical_support(self):
        model, anchor = predictive_fixture()
        model = model.install_details_from_coarse([0])
        reference = {k:v.clone()+.31 for k,v in anchor.items()}
        quantized = model.reincarnation_quantized_raw(config(), 1., reference, torch.zeros(len(model._xyz), dtype=torch.long))
        raw = model_raw(model)
        for name in ('xyz', 'rotation', 'logscale', 'opacity'):
            self.assertTrue(equal_bytes(raw[name][2:], quantized[name][2:]), name)

    def test_phase2_novel_geometry_remains_fixed_after_real_adam_momentum_and_restore(self):
        coarse, _ = predictive_fixture()
        model = coarse.install_details_from_coarse([0])
        model.training_setup(args())
        objective(model).backward(); model.optimizer.step(); model.optimizer.zero_grad(set_to_none=True)
        for attr in ('_xyz', '_rotation', '_scaling'):
            self.assertGreater(float(model.optimizer.state[getattr(model, attr)]['exp_avg'][:2].abs().sum()), 0.)
        model.set_training_stage(2)
        locked = {attr:getattr(model, attr)[:2].detach().clone() for attr in ('_xyz', '_rotation', '_scaling')}
        restored = PredictiveReincarnationModel(1).restore(model.capture(), args())
        for branch in (model, restored):
            objective(branch).backward(); branch.optimizer.step(); branch.optimizer.zero_grad(set_to_none=True)
            for attr, before in locked.items():
                self.assertTrue(equal_bytes(getattr(branch, attr)[:2], before), attr)
                self.assertTrue(bool((branch.optimizer.state[getattr(branch, attr)]['exp_avg'][:2] == 0).all()))

    def test_phase1_new_rows_keep_birth_alpha_while_learning_then_phase2_can_exit(self):
        coarse, anchor = predictive_fixture()
        model = coarse.install_details_from_coarse([0], detail_alpha=.13)
        model.training_setup(args())
        protected, floor = model._maturity_opacity_floor()
        for _ in range(5):
            model.reincarnation_phenotype()['opacity'][protected].sum().backward()
            model.optimizer.step(); model.optimizer.zero_grad(set_to_none=True); model.project_parameters()
            actual = model.reincarnation_quantized_raw(config(), 1., anchor,
                torch.zeros(len(model._xyz), dtype=torch.long))['opacity'][protected]
            self.assertTrue(bool((actual >= floor).all()))
        restored = PredictiveReincarnationModel(1).restore(model.capture(), args())
        self.assertEqual(restored.reincarnation_specs[0]['detail_birth_alpha'], .13)
        restored.set_training_stage(2)
        restored.reincarnation_phenotype()['opacity'][protected].sum().backward()
        restored.optimizer.step(); restored.optimizer.zero_grad(set_to_none=True); restored.project_parameters()
        self.assertTrue(bool((model_raw(restored)['opacity'][protected] < floor).all()))


class PredictiveEndpointTests(unittest.TestCase):
    def mature(self):
        model, _ = predictive_fixture()
        model.p_common_dc.data.add_(.08)
        model.p_common_sh1.data.add_(.02)
        model = model.install_details_from_coarse([0]).set_training_stage(2)
        model._features_dc.data[model.detail_rows].add_(.2)
        model._features_rest.data[model.detail_rows].sub_(.03)
        model._features_dc.data[:2].add_(.1)
        model.project_parameters()
        return model

    def assert_bounded(self, model, raw):
        low, high = model._radiance_bounds()
        actual_low, actual_high = sh1_sphere_range(raw['sh_dc'], raw['sh_rest'])
        self.assertTrue(bool((actual_low >= low-2.e-14).all()))
        self.assertTrue(bool((actual_high <= high+2.e-14).all()))

    def test_whole_p_prior_and_pure_dc_sh_endpoints_keep_novel_current_and_support(self):
        model = self.mature().freeze_predictive_endpoints()
        endpoints = {kind:model.endpoint_raw(kind) for kind in ('prior', 'dc', 'sh', 'full')}
        dc0, rest0 = model._prior_features()
        p = slice(model.novel_count, None)
        torch.testing.assert_close(endpoints['prior']['sh_dc'][p], dc0[p], rtol=0, atol=0)
        torch.testing.assert_close(endpoints['prior']['sh_rest'][p], rest0[p], rtol=0, atol=0)
        torch.testing.assert_close(endpoints['dc']['sh_rest'][p], rest0[p], rtol=0, atol=0)
        torch.testing.assert_close(endpoints['sh']['sh_dc'][p], dc0[p], rtol=0, atol=0)
        for kind, raw in endpoints.items():
            self.assert_bounded(model, raw)
            for name in ('xyz', 'rotation', 'logscale', 'opacity'):
                self.assertTrue(equal_bytes(raw[name], endpoints['full'][name]), (kind, name))
            for name in ('sh_dc', 'sh_rest'):
                self.assertTrue(equal_bytes(raw[name][:2], endpoints['full'][name][:2]), (kind, name))
        endpoints['prior']['sh_dc'].add_(100.)
        self.assertTrue(bool((model.endpoint_raw('prior')['sh_dc'].abs() < 100).all()))

    def test_convex_apply_is_nonratcheting_inplace_and_does_not_touch_optimizer(self):
        model = self.mature()
        model.training_setup(args())
        objective(model).backward(); model.optimizer.step(); model.optimizer.zero_grad(set_to_none=True)
        model.project_parameters(); model.freeze_predictive_endpoints()
        objects = [id(getattr(model, attr)) for attr in RAW_ATTRIBUTES]+[id(model.p_common_dc), id(model.p_common_sh1)]
        states = {parameter:{k:v.clone() if torch.is_tensor(v) else v for k,v in state.items()}
                  for parameter,state in model.optimizer.state.items()}
        support = model_raw(model)
        model.apply_predictive_weights([.2, .3, .4])
        first = model_raw(model)
        model.apply_predictive_weights([0., 0., 0.])
        model.apply_predictive_weights([.2, .3, .4])
        second = model_raw(model)
        self.assertEqual(objects, [id(getattr(model, attr)) for attr in RAW_ATTRIBUTES]+[id(model.p_common_dc), id(model.p_common_sh1)])
        for name in RAW_FIELDS:
            self.assertTrue(equal_bytes(first[name], second[name]), name)
            if name not in ('sh_dc', 'sh_rest'):
                self.assertTrue(equal_bytes(support[name], second[name]), name)
        self.assert_bounded(model, second)
        for parameter,state in states.items():
            for key,value in state.items():
                if torch.is_tensor(value):
                    self.assertTrue(equal_bytes(value, model.optimizer.state[parameter][key]))
        self.assertFalse(model.p_common_dc.requires_grad)
        before = model_raw(model)
        with self.assertRaises(ValueError): model.apply_predictive_weights([.8, .8, 0.])
        for name in RAW_FIELDS:
            self.assertTrue(equal_bytes(before[name], model_raw(model)[name]))

    def test_real_hdr_prior_bounds_children_and_is_independent_of_codec_predictor(self):
        coarse, _ = predictive_fixture(n=0)
        coarse.support_anchor_raw['sh_dc'][0].fill_(4.)
        coarse.p_common_dc.data[0].fill_(4.)
        model = coarse.install_details_from_coarse([0])
        immutable = model.support_anchor_raw['sh_dc'].clone()
        model._features_dc.data.fill_(100.)
        model._features_rest.data.fill_(50.)
        model.p_common_dc.data.fill_(100.)
        model.project_parameters()
        raw = model_raw(model)
        self.assert_bounded(model, raw)
        low, high = sh1_sphere_range(raw['sh_dc'], raw['sh_rest'])
        self.assertGreater(float(high[model.detail_rows].max()), 1.)
        self.assertTrue(equal_bytes(model.support_anchor_raw['sh_dc'], immutable))
        model.freeze_predictive_endpoints()
        for kind in ('prior', 'dc', 'sh', 'full'):
            self.assert_bounded(model, model.endpoint_raw(kind))


class PredictiveStateTests(unittest.TestCase):
    def test_expansion_preserves_parent_adam_and_immutable_novel_seed(self):
        coarse, _ = predictive_fixture()
        coarse.training_setup(args())
        objective(coarse).backward(); coarse.optimizer.step(); coarse.optimizer.zero_grad(set_to_none=True)
        expected = coarse.optimizer.state[coarse.p_common_dc]['exp_avg'].clone()
        seed = {k:v.clone() for k,v in coarse.novel_initial_raw.items()}
        model = coarse.install_details_from_coarse([0])
        self.assertTrue(equal_bytes(expected, model.optimizer.state[model.p_common_dc]['exp_avg']))
        for name in RAW_FIELDS:
            self.assertTrue(equal_bytes(model.novel_initial_raw[name], seed[name]))
        self.assertEqual(model.predictive_phase, 1)
        self.assertFalse(model.p_common_dc.requires_grad)

    def test_capture_restore_and_atomic_subset_keep_endpoints_phase_and_adam(self):
        model, _ = predictive_fixture()
        model = model.install_details_from_coarse([0]).set_training_stage(2)
        model.training_setup(args())
        objective(model).backward(); model.optimizer.step(); model.optimizer.zero_grad(set_to_none=True)
        model.project_parameters(); model.freeze_predictive_endpoints(); model.apply_predictive_weights([.2, .1, .4])
        packet = model.capture()
        self.assertIn('predictive_reincarnation', packet)
        restored = PredictiveReincarnationModel(1).restore(packet, args())
        self.assertEqual(restored.predictive_phase, 2)
        for name in RAW_FIELDS:
            self.assertTrue(equal_bytes(model_raw(model)[name], model_raw(restored)[name]), name)
        with self.assertRaisesRegex(ValueError, 'atomically'):
            restored.reincarnation_subset([0, 2, 3])
        ids = torch.tensor([1, 2, 3, 4])
        selected = restored.reincarnation_subset(ids)
        self.assertEqual(selected.novel_count, 1)
        self.assertEqual(selected.reincarnation_groups[0].tolist(), [1, 2, 3])
        self.assertEqual(selected.predictive_phase, 2)
        for kind in ('prior', 'dc', 'sh', 'full'):
            for name in RAW_FIELDS:
                self.assertTrue(equal_bytes(selected.endpoint_raw(kind)[name], restored.endpoint_raw(kind)[name][ids]))
        for attr in ('_opacity', '_features_dc'):
            old, new = getattr(restored, attr), getattr(selected, attr)
            self.assertTrue(equal_bytes(selected.optimizer.state[new]['exp_avg'], restored.optimizer.state[old]['exp_avg'][ids]))
        selected.apply_predictive_weights([0., 0., 1.])
        self.assertFalse(equal_bytes(model_raw(selected)['sh_dc'], model_raw(restored)['sh_dc'][ids]))

    def test_invalid_stages_roots_and_legacy_mass_offsets_are_explicit(self):
        model, _ = predictive_fixture()
        with self.assertRaises(ValueError): model.set_training_stage(3)
        with self.assertRaises(ValueError): model.install_details_from_coarse([0, 0])
        with self.assertRaises(ValueError): model.install_details_from_coarse([0], detail_alpha=1.)
        with self.assertRaises(ValueError): model.install_families_from_coarse([0], tangent_offsets=torch.zeros(1, 2))
        with self.assertRaises(NotImplementedError): PredictiveReincarnationModel(3)


# ----- test_dual_predictive_controller -----
M=importlib.import_module('pipeline.module')


class ImageRenderer:
 """Small differentiable image adapter; actual native backend has separate proof."""
 collect_contributions=False
 def __init__(self,model):self.gaussians=model
 def __call__(self,view,background):
  value=self.gaussians
  colour=.5+.28209479177387814*value.get_features[:,0]
  alpha=value.get_opacity[:,0]
  rgb=(colour*alpha[:,None]).sum(0)/alpha.sum().clamp_min(1e-8)
  return {'render':rgb[:,None,None].expand(3,4,4)}


def dual_fixture():
 m,_=base_fixture();m.config.birth.radiance_mode='dual_predictive_reincarnation'
 m.config.rigid_tree.birth_steps=600
 views=m.transaction_camera_views()
 for name,v in views.items():v.path=name;v.image=torch.full((3,4,4),.55)
 original=model_raw(m.gaussians)
 specs=[dict(old_row=2,raw={key:value[2].clone() for key,value in original.items()},
             responsibility=1.,split_eligible=True,tangent_offset=torch.tensor([.01,.01]))]
 novel={key:value[:0].clone() for key,value in original.items()}
 model=PredictiveReincarnationModel(1).install_reincarnations(specs,novel_raw=novel)
 names=tuple(views);template=NS(group_ids=torch.empty(0,dtype=torch.long),scores=torch.empty(0),
   support_domains=[],fit_names=names,validation_names=names)
 state=dict(camera_groups={'A':names[::2],'B':names[1::2]},novel_initial_raw=novel,eligible=False)
 tx=rb._reincarnation_transaction(m,model,specs,template,state,{});tx.predictive_model=True
 m.birth_cohort=0;m.birth_cohort_start=200;m.birth_end_step=800;m.frame_finalized=False;m.motion_committed=True
 m.background=torch.zeros(3);m.renderer=ImageRenderer(m.gaussians)
 for name in ('_install_birth_transaction','_capture_birth_pool','_activate_dual_branch','_restore_birth_pool'):
  setattr(m,name,lambda *args,_name=name,**kwargs:getattr(M.Module,_name)(m,*args,**kwargs))
 M.Module._install_birth_transaction(m,tx)
 baseline=rb.encode_trial(m,torch.zeros(1,dtype=torch.bool),torch.zeros(len(m.gaussians._xyz),dtype=torch.bool))
 m.birth_transaction.old_render_reference=baseline.raw
 M.Module._begin_dual_branches(m)
 return m,views


class DualControllerTests(unittest.TestCase):
 def test_real_codec_calibration_proves_same_selected_population(self):
  m,views=dual_fixture();m._activate_dual_branch('A')
  with torch.no_grad():m.births.p_common_dc.fill_(.4)
  mask=torch.ones(len(m.births._xyz),dtype=torch.bool);deleted=torch.zeros(len(m.gaussians._xyz),dtype=torch.bool);deleted[2]=True
  target=rb.encode_trial(m,mask,deleted)
  previous=m.renderer.gaussians;m.renderer.gaussians=rb.RawView(target.raw,1,m.gaussians)
  for view in views.values():view.image=m.renderer(view,m.background)['render'].detach().clone()
  m.renderer.gaussians=previous
  old=model_raw(m.gaussians)
  trial,report=M.Module._calibrate_dual_population(m,'A',mask,deleted)
  self.assertTrue(report['certificate']);self.assertGreater(sum(report['weights']),.9)
  self.assertEqual(report['population_count'],1);self.assertEqual(report['survivor_count'],len(m.gaussians._xyz)-1)
  self.assertEqual(report['endpoint_scope'],'P_whole_family_only_N_current_constant')
  replay=rb.encode_trial(m,mask,deleted)
  for key in RAW_FIELDS:
   torch.testing.assert_close(replay.raw[key],trial.raw[key],atol=0,rtol=0)
   torch.testing.assert_close(model_raw(m.gaussians)[key],old[key],atol=0,rtol=0)
  self.assertTrue(all(x['improves_incumbent'] for x in report['folds'].values()))

 def test_both_branches_checkpoint_complete_optimizer_without_alias(self):
  m,_=dual_fixture();m.birth_branches['A']['model'].p_common_dc.grad=torch.ones_like(m.births.p_common_dc)
  m.birth_branches['A']['model'].optimizer.step();m.birth_branches['A']['model'].optimizer.zero_grad(set_to_none=True)
  snapshot=M.Module._capture_dual_checkpoint(m)
  a=m.birth_branches['A']['model'].p_common_dc.detach().clone();b=m.birth_branches['B']['model'].p_common_dc.detach().clone()
  M.Module._restore_dual_checkpoint(m,snapshot)
  torch.testing.assert_close(m.birth_branches['A']['model'].p_common_dc,a,atol=0,rtol=0)
  torch.testing.assert_close(m.birth_branches['B']['model'].p_common_dc,b,atol=0,rtol=0)
  self.assertGreater(len(m.birth_branches['A']['model'].optimizer.state),0)
  self.assertEqual(len(m.birth_branches['B']['model'].optimizer.state),0)
  with torch.no_grad():m.birth_branches['A']['model'].p_common_dc.add_(1)
  torch.testing.assert_close(m.birth_branches['B']['model'].p_common_dc,b,atol=0,rtol=0)
  torch.testing.assert_close(snapshot['branches']['A']['model']['predictive_reincarnation']['common_dc'],a,atol=0,rtol=0)

 def test_phase_transition_has_fresh_adam_and_preserves_N_initialization(self):
  m,_=dual_fixture();m.dual_training_totals={}
  M.Module._advance_dual_stage(m,800)
  self.assertEqual(m.birth_cohort,1);self.assertEqual(m.birth_end_step,1400)
  for packet in m.birth_branches.values():
   self.assertEqual(packet['model'].predictive_phase,1)
   self.assertEqual(len(packet['model'].reincarnation_groups[0]),3)
   self.assertEqual(len(packet['model'].optimizer.state),0)
  M.Module._advance_dual_stage(m,1400)
  self.assertEqual(m.birth_cohort,2);self.assertEqual(m.birth_end_step,2000)
  for packet in m.birth_branches.values():
   self.assertEqual(packet['model'].predictive_phase,2)
   self.assertEqual(len(packet['model'].optimizer.state),0)
   self.assertFalse(packet['model'].p_common_dc.requires_grad)


# ----- test_source_residual_model -----
def same(a,b):
    return a.detach().cpu().contiguous().numpy().tobytes()==b.detach().cpu().contiguous().numpy().tobytes()


def source_model_fixture(n=1):
    anchor=_empty_raw(3,1,'cpu',torch.float64)
    anchor['xyz'][:]=torch.tensor([[.1,-.1,3.],[-.2,.2,3.3],[.2,.1,3.7]],dtype=torch.float64)
    anchor['logscale'][:]=torch.tensor([[.2,.1,.01],[.15,.09,.008],[.12,.07,.01]],dtype=torch.float64).log()
    anchor['opacity'][:]=torch.tensor([[.2837],[1.073],[.83]],dtype=torch.float64)
    anchor['sh_dc'][:]=torch.tensor([[[-0.,.2,-.3]],[[.1,-.2,.3]],[[.2,.1,.3]]],dtype=torch.float64)
    anchor['sh_rest'][:]=torch.tensor([[[.03,-.02,.01],[.04,.02,-.03],[-.02,.01,.04]]],dtype=torch.float64)
    novel=_empty_raw(n,1,'cpu',torch.float64);novel['xyz'][:,2]=3.1
    novel['opacity'].fill_(-2.);novel['logscale'].fill_(-3.)
    specs=[dict(old_row=i,raw={k:v[i].clone() for k,v in anchor.items()}) for i in range(3)]
    m=SourceResidualModel(1).install_reincarnations(specs,novel_raw=novel)
    m.spatial_lr_scale=1.
    return m,anchor


def fine_fixture():
    coarse,anchor=source_model_fixture()
    coarse.p_common_dc.data.add_(.2)
    coarse.p_common_sh1.data.add_(.03)
    coarse.project_parameters()
    fine=coarse.install_details_from_coarse([0,2],detail_alpha=.12)
    return coarse,fine,anchor


def loss(m):
    return sum(value.sum() for value in m.reincarnation_phenotype().values())


class SourceBackgroundTests(unittest.TestCase):
    def test_coarse_native_identity_and_legacy_four_labels(self):
        m,anchor=source_model_fixture()
        self.assertEqual(m.endpoint_kinds(),('prior','dc','sh','full'))
        raw=m.reincarnation_quantized_raw(config(),1.,anchor,torch.zeros(len(m._xyz),dtype=torch.long))
        for name in RAW_FIELDS:self.assertTrue(same(raw[name][1:],anchor[name]),name)
        self.assertTrue(bool((raw['opacity'][:1].sigmoid()==0).all()))
        m.p_common_dc.data.add_(.05)
        m.freeze_predictive_endpoints();m.apply_predictive_weights([0.,0.,1.])
        self.assertFalse(same(model_raw(m)['sh_dc'][1:],anchor['sh_dc']))

    def test_install_saves_C_but_resets_mother_and_every_child_to_immutable_source(self):
        coarse,m,anchor=fine_fixture()
        self.assertEqual(m.endpoint_kinds(),('prior','common','detail','full'))
        self.assertTrue(same(m.common_material['sh_dc'],coarse.p_common_dc))
        self.assertTrue(same(m.common_material['sh_rest'],coarse.p_common_sh1))
        self.assertTrue(same(m.p_common_dc,anchor['sh_dc']))
        self.assertTrue(same(m.p_common_sh1,anchor['sh_rest']))
        raw=model_raw(m)
        for root,rows in enumerate(m.reincarnation_groups):
            for name in ('sh_dc','sh_rest'):
                self.assertTrue(same(raw[name][rows],anchor[name][root:root+1].expand(len(rows),*anchor[name].shape[1:])),name)
            for name in ('xyz','rotation','logscale','opacity'):
                self.assertTrue(same(raw[name][rows[0]],anchor[name][root]),name)
        for name in RAW_FIELDS:
            self.assertTrue(same(m.novel_initial_raw[name],coarse.novel_initial_raw[name]))

    def test_two_fine_phases_have_real_child_gradients_but_mother_material_stays_theta0(self):
        _,m,anchor=fine_fixture();m.training_setup(args())
        common={k:v.clone() for k,v in m.common_material.items()}
        for phase in (1,2):
            m.set_training_stage(phase)
            for _ in range(3):
                raw=m.reincarnation_phenotype();loss(m).backward()
                self.assertIsNone(m.p_common_dc.grad);self.assertIsNone(m.p_common_sh1.grad)
                self.assertGreater(float(m._features_dc.grad[m.detail_rows].abs().sum()),0.)
                self.assertGreater(float(m._features_rest.grad[m.detail_rows].abs().sum()),0.)
                self.assertGreater(float(m._opacity.grad[m.detail_rows].abs().sum()),0.)
                m.optimizer.step();m.optimizer.zero_grad(set_to_none=True);m.project_parameters()
                self.assertTrue(same(m.p_common_dc,anchor['sh_dc']))
                self.assertTrue(same(m.p_common_sh1,anchor['sh_rest']))
        for name in common:self.assertTrue(same(common[name],m.common_material[name]))

    def test_zero_details_are_original_source_identity_not_mature_C(self):
        _,m,anchor=fine_fixture();zero=m.zero_details_raw()
        for name in RAW_FIELDS:
            self.assertTrue(same(zero[name][m.reincarnation_rows],anchor[name]),name)
        self.assertTrue(bool((zero['opacity'][m.detail_rows].sigmoid()==0).all()))

    def test_true_pristine_C_D_preserve_source_bits_in_all_fine_endpoints(self):
        coarse,_=source_model_fixture();m=coarse.install_details_from_coarse([0,2])
        m.freeze_predictive_endpoints();prior=m.endpoint_raw('prior')
        for kind in m.endpoint_kinds():
            for name in RAW_FIELDS:self.assertTrue(same(m.endpoint_raw(kind)[name],prior[name]),(kind,name))

    def test_float32_hdr_boundary_mother_keeps_native_bits_with_saved_C_and_live_D(self):
        raw=_empty_raw(1,1,'cpu',torch.float32)
        raw['xyz'][0,2]=3.;raw['logscale'].fill_(-3.)
        raw['sh_dc'][0]=torch.tensor([[-.17231953144073486,-.49404096603393555,-1.159371018409729]])
        raw['sh_rest'][0]=torch.tensor([[.21131521463394165,.27519354224205017,.3959043622016907],
            [-.053831297904253006,-.03557174280285835,-.0494646281003952],
            [.09111939370632172,.023353423923254013,-.0006613947916775942]])
        m=SourceResidualModel(1).install_reincarnations([dict(old_row=0,raw={k:v[0] for k,v in raw.items()})])
        m.p_common_dc.data.add_(.1);m.project_parameters()
        m=m.install_details_from_coarse([0])
        m._features_dc.data[m.detail_rows] += .15
        for _ in range(3):m.project_parameters()
        self.assertFalse(same(m.common_material['sh_dc'],raw['sh_dc']))
        self.assertFalse(same(model_raw(m)['sh_dc'][m.detail_rows],raw['sh_dc'].expand(2,1,3)))
        self.assertTrue(same(m.p_common_dc,raw['sh_dc']))
        self.assertTrue(same(m.p_common_sh1,raw['sh_rest']))


class IndependentEndpointTests(unittest.TestCase):
    def learned(self):
        _,m,anchor=fine_fixture();m.set_training_stage(2)
        m._features_dc.data[m.detail_rows] += .15
        m._features_rest.data[m.detail_rows] -= .012
        m._features_dc.data[:1].add_(.08)
        m.project_parameters();m.freeze_predictive_endpoints()
        return m,anchor

    def test_endpoint_common_and_detail_are_independent_whole_family_controls(self):
        m,anchor=self.learned()
        p,c,d,f=(m.endpoint_raw(k) for k in m.endpoint_kinds())
        n=m.novel_count
        for root,rows in enumerate(m.reincarnation_groups):
            for name in ('sh_dc','sh_rest'):
                self.assertTrue(same(p[name][rows],anchor[name][root:root+1].expand(len(rows),*anchor[name].shape[1:])))
                self.assertTrue(same(c[name][rows],m.common_material[name][root:root+1].expand(len(rows),*anchor[name].shape[1:])))
                self.assertTrue(same(d[name][rows[0]],anchor[name][root]))
        self.assertFalse(same(d['sh_dc'][m.detail_rows],p['sh_dc'][m.detail_rows]))
        rawdc=c['sh_dc']+(d['sh_dc']-p['sh_dc']);rawrest=c['sh_rest']+(d['sh_rest']-p['sh_rest'])
        low,high=m._radiance_bounds();dc,rest,_=project_sh1_sphere(rawdc,rawrest,low,high)
        torch.testing.assert_close(f['sh_dc'][n:],dc[n:],rtol=0,atol=0)
        torch.testing.assert_close(f['sh_rest'][n:],rest[n:],rtol=0,atol=0)
        for raw in (p,c,d,f):
            for name in ('xyz','rotation','logscale','opacity'):self.assertTrue(same(raw[name],p[name]))
            for name in ('sh_dc','sh_rest'):self.assertTrue(same(raw[name][:n],p[name][:n]))
            lo,hi=sh1_sphere_range(raw['sh_dc'],raw['sh_rest'])
            self.assertTrue(bool((lo>=low-2.e-14).all() & (hi<=high+2.e-14).all()))

    def test_reject_common_can_keep_nonzero_detail_and_final_common_can_update_mother(self):
        m,anchor=self.learned();objects=[id(getattr(m,a)) for a in RAW_ATTRIBUTES]+[id(m.p_common_dc),id(m.p_common_sh1)]
        detail=m.endpoint_raw('detail')
        m.apply_predictive_weights([0.,1.,0.])
        raw=model_raw(m)
        for name in ('sh_dc','sh_rest'):
            self.assertTrue(same(raw[name][m.reincarnation_rows],anchor[name]))
            torch.testing.assert_close(raw[name],detail[name],rtol=0,atol=2.e-16)
        self.assertFalse(same(raw['sh_dc'][m.detail_rows],m.endpoint_raw('prior')['sh_dc'][m.detail_rows]))
        m.apply_predictive_weights([1.,0.,0.])
        self.assertTrue(same(m.p_common_dc,m.common_material['sh_dc']))
        self.assertTrue(same(m.p_common_sh1,m.common_material['sh_rest']))
        self.assertEqual(objects,[id(getattr(m,a)) for a in RAW_ATTRIBUTES]+[id(m.p_common_dc),id(m.p_common_sh1)])
        m.apply_predictive_weights([0.,1.,0.]);again=model_raw(m)
        for name in RAW_FIELDS:self.assertTrue(same(raw[name],again[name]))

    def test_extreme_C_plus_D_is_source_bounded_without_changing_original_hdr_prior(self):
        m,anchor=source_model_fixture(n=0)
        m.support_anchor_raw['sh_dc'][0].fill_(4.);m.p_common_dc.data[0].fill_(4.)
        m.p_common_dc.data.add_(20.);m.p_common_sh1.data.add_(10.);m.project_parameters()
        m=m.install_details_from_coarse([0])
        m._features_dc.data[m.detail_rows] += 100.;m._features_rest.data[m.detail_rows] += 20.
        m.project_parameters();m.freeze_predictive_endpoints()
        low,high=m._radiance_bounds()
        for k in m.endpoint_kinds():
            raw=m.endpoint_raw(k);lo,hi=sh1_sphere_range(raw['sh_dc'],raw['sh_rest'])
            self.assertTrue(bool((lo>=low-2.e-14).all() & (hi<=high+2.e-14).all()))
        self.assertGreater(float(high[m.detail_rows].max()),1.)
        self.assertEqual(float(m.support_anchor_raw['sh_dc'][0,0,0]),4.)


class SourceResidualStateTests(unittest.TestCase):
    def test_capture_restore_subset_map_C_endpoints_and_adam_without_alias(self):
        _,m,anchor=fine_fixture();m.training_setup(args())
        loss(m).backward();m.optimizer.step();m.optimizer.zero_grad(set_to_none=True);m.project_parameters()
        m.set_training_stage(2);m.freeze_predictive_endpoints();m.apply_predictive_weights([.2,.6,.1])
        packet=m.capture();self.assertIn('source_residual',packet)
        r=SourceResidualModel(1).restore(packet,args())
        self.assertEqual(r.endpoint_kinds(),m.endpoint_kinds())
        for name in RAW_FIELDS:self.assertTrue(same(model_raw(r)[name],model_raw(m)[name]))
        self.assertTrue(same(r.common_material['sh_dc'],m.common_material['sh_dc']))
        ids=torch.cat((torch.tensor([0]),m.reincarnation_groups[2]))
        subset=r.reincarnation_subset(ids)
        self.assertTrue(same(subset.common_material['sh_dc'],r.common_material['sh_dc'][2:3]))
        self.assertTrue(same(subset.common_material['sh_rest'],r.common_material['sh_rest'][2:3]))
        for k in m.endpoint_kinds():
            for name in RAW_FIELDS:self.assertTrue(same(subset.endpoint_raw(k)[name],r.endpoint_raw(k)[name][ids]))
        for attr in ('_features_dc','_opacity'):
            a,b=getattr(r,attr),getattr(subset,attr)
            self.assertTrue(same(subset.optimizer.state[b]['exp_avg'],r.optimizer.state[a]['exp_avg'][ids]))
        restored_subset=SourceResidualModel(1).restore(subset.capture(),args())
        self.assertTrue(same(restored_subset.common_material['sh_dc'],m.common_material['sh_dc'][2:3]))
        self.assertTrue(same(restored_subset.common_material['sh_rest'],m.common_material['sh_rest'][2:3]))
        for kind in m.endpoint_kinds():
            for name in RAW_FIELDS:
                self.assertTrue(same(restored_subset.endpoint_raw(kind)[name],r.endpoint_raw(kind)[name][ids]))
        subset.common_material['sh_dc'].add_(.1)
        self.assertFalse(same(subset.common_material['sh_dc'],r.common_material['sh_dc'][2:3]))
        with self.assertRaisesRegex(ValueError,'atomically'):r.reincarnation_subset([0,int(r.detail_rows[0])])

    def test_saving_C_does_not_make_a_new_prior_on_repeated_detail_install(self):
        coarse,m,anchor=fine_fixture();saved={k:v.clone() for k,v in m.common_material.items()}
        m._features_dc.data[m.detail_rows] += .1
        expanded=m.install_details_from_coarse([1])
        for name,value in saved.items():self.assertTrue(same(expanded.common_material[name],value))
        self.assertTrue(same(expanded.support_anchor_raw['sh_dc'],anchor['sh_dc']))
        self.assertTrue(same(expanded.p_common_dc,anchor['sh_dc']))


# ----- test_source_residual_controller -----
class SpatialImageRenderer:
 collect_contributions=False
 def __init__(self,model):self.gaussians=model
 def __call__(self,view,background):
  m=self.gaussians;xyz=m.get_xyz;scale=m.get_scaling
  y,x=torch.meshgrid(torch.linspace(-.15,.65,8),torch.linspace(-.15,.65,8),indexing='ij')
  dist=((x[None]-xyz[:,0,None,None])/scale[:,0,None,None]).square()+((y[None]-xyz[:,1,None,None])/scale[:,1,None,None]).square()
  alpha=m.get_opacity[:,0,None,None]*torch.exp(-.5*dist)
  T=torch.cumprod(torch.cat((torch.ones_like(alpha[:1]),1-alpha),0),0)[:-1]
  colour=.5+.28209479177387814*m.get_features[:,0]
  image=torch.einsum('nhw,nc->chw',alpha*T,colour)
  return {'render':image}


def source_controller_fixture():
 m,views=dual_fixture();m.config.birth.radiance_mode='source_residual'
 tx=m.birth_transaction
 source=SourceResidualModel(1).install_reincarnations(tx.reincarnation_specs,
      novel_raw=tx.crossfit_state['novel_initial_raw'])
 tx.reincarnation_state=source.reincarnation_state();tx.raw=model_raw(source)
 M.Module._install_birth_transaction(m,tx);M.Module._begin_dual_branches(m)
 m.renderer=SpatialImageRenderer(m.gaussians)
 return m,views


class SourceResidualControllerTests(unittest.TestCase):
 def test_fine_factory_restarts_source_material_and_restores_optional_C(self):
  m,_=source_controller_fixture()
  for packet in m.birth_branches.values():
   with torch.no_grad():packet['model'].p_common_dc.fill_(.7)
  M.Module._advance_dual_stage(m,800)
  for packet in m.birth_branches.values():
   model=packet['model'];self.assertIsInstance(model,SourceResidualModel)
   self.assertEqual(model.endpoint_kinds(),('prior','common','detail','full'))
   torch.testing.assert_close(model.p_common_dc,model.support_anchor_raw['sh_dc'],atol=0,rtol=0)
   torch.testing.assert_close(model.common_material['sh_dc'],torch.full_like(model.p_common_dc,.7),atol=0,rtol=0)
  snapshot=M.Module._capture_dual_checkpoint(m)
  M.Module._restore_dual_checkpoint(m,snapshot)
  for packet in m.birth_branches.values():
   self.assertIsInstance(packet['model'],SourceResidualModel)
   torch.testing.assert_close(packet['model'].common_material['sh_dc'],torch.full_like(packet['model'].p_common_dc,.7),atol=0,rtol=0)

 def test_actual_codec_colour_calibrator_can_keep_D_without_C(self):
  m,views=source_controller_fixture()
  for packet in m.birth_branches.values():
   with torch.no_grad():packet['model'].p_common_dc.fill_(.7)
  M.Module._advance_dual_stage(m,800);m._activate_dual_branch('A')
  with torch.no_grad():m.births._features_dc[m.births.detail_rows]=.5
  m.births.freeze_predictive_endpoints()
  m.births.apply_predictive_weights([0.,1.,0.])
  mask=torch.ones(len(m.births._xyz),dtype=torch.bool)
  deleted=torch.zeros(len(m.gaussians._xyz),dtype=torch.bool);deleted[2]=True
  target=rb.encode_trial(m,mask,deleted)
  previous=m.renderer.gaussians;m.renderer.gaussians=rb.RawView(target.raw,1,m.gaussians)
  for view in views.values():view.image=m.renderer(view,m.background)['render'].detach().clone()
  m.renderer.gaussians=previous
  trial,report=M.Module._calibrate_dual_population(m,'A',mask,deleted)
  self.assertEqual(report['endpoint_order'],['prior','common','detail','full'])
  self.assertGreater(report['weights_by_endpoint']['detail'],.99)
  self.assertLess(report['weights_by_endpoint']['common'],1e-5)
  self.assertTrue(report['certificate'])
  replay=rb.encode_trial(m,mask,deleted)
  for name in RAW_FIELDS:torch.testing.assert_close(replay.raw[name],trial.raw[name],atol=0,rtol=0)


# ----- test_shadow_birth_initializer -----
class ShadowDensityMathTests(unittest.TestCase):
    def test_three_raw_images_reconstruct_exact_quartic(self):
        gen=torch.Generator().manual_seed(39)
        i=torch.rand(3,4,5,dtype=torch.float64,generator=gen)
        a=torch.randn(3,4,5,dtype=torch.float64,generator=gen)*.1
        b=torch.randn(3,4,5,dtype=torch.float64,generator=gen)*.2
        target=torch.rand(3,4,5,dtype=torch.float64,generator=gen)
        c=density_coefficients(i,i+.5*a+.25*b,i+a+b,target)
        for rho in (0.,.13,.49,.85,1.):
            expected=(i+a*rho+b*rho*rho-target).square().mean()
            got=sum(c[k]*rho**k for k in range(5))
            torch.testing.assert_close(got,expected,rtol=0,atol=2e-16)

    def test_linear_image_degenerate_cubic_recovers_exact_density(self):
        rho,r=solve_density_quartic([.04,-.4,1.,0.,0.])
        self.assertAlmostEqual(rho,.2,places=12)
        self.assertLess(abs(r['proposal_surrogate_mse']),1e-15)

    def test_two_minima_and_endpoint_prior_are_global_not_local_choices(self):
        # E=(rho^2-rho+.16)^2, equal minima at .2 and .8.
        rho,r=solve_density_quartic([.0256,-.32,1.32,-2.,1.])
        self.assertAlmostEqual(rho,.2,places=10)
        self.assertLess(abs(r['proposal_surrogate_mse']),1e-15)
        rho,_=solve_density_quartic([.1,1.,1.,1.,1.])
        self.assertEqual(rho,0.)
        rho,_=solve_density_quartic([1.,-2.,1.,0.,0.])
        self.assertAlmostEqual(rho,1.,places=12)

    def test_multiple_root_and_zero_response_are_stable(self):
        rho,r=solve_density_quartic([.0625,-.5,1.5,-2.,1.]) # (rho-.5)^4
        self.assertAlmostEqual(rho,.5,places=4)
        self.assertLess(abs(r['proposal_surrogate_mse']),1e-14)
        rho,_=solve_density_quartic([.3,0.,0.,0.,0.])
        self.assertEqual(rho,0.)

    def test_constant_baseline_does_not_swallow_small_density_benefit(self):
        rho,_=solve_density_quartic([100.,-4e-12,1e-11,0.,0.])
        self.assertAlmostEqual(rho,.2,places=12)

    def test_nonfinite_coefficients_and_image_shapes_are_rejected(self):
        with self.assertRaises(ValueError):solve_density_quartic([0.,0.,float('nan'),0.,0.])
        p=torch.zeros(3,2,2)
        with self.assertRaises(ValueError):density_coefficients(p,p[:,:1],p,p)

    def test_real_alpha_compositing_with_old_layer_between_children_is_quadratic(self):
        # Fixed-depth alpha compositing, including an old layer between the
        # siblings; all original layers remain untouched at rho=0.
        old=torch.tensor([[.2,.4,.3,.1],[.3,.2,.4,.3],[.8,.7,.6,.9]],dtype=torch.float64)
        child=torch.tensor([[.7,.4,.1,.6],[.2,.8,.4,.3]],dtype=torch.float64)
        colour=torch.tensor([[.3,.5,.7],[.8,.1,.4],[.2,.7,.4],[.6,.3,.2],[.4,.6,.8]],dtype=torch.float64)
        def render(rho):
            a=torch.stack((old[0],rho*child[0],old[1],rho*child[1],old[2]))
            trans=torch.cumprod(torch.cat((torch.ones_like(a[:1]),1-a),0),0)[:-1]
            return torch.einsum('np,nc->cp',a*trans,colour)[:,None]
        target=render(.3);coeff=density_coefficients(render(0),render(.5),render(1),target)
        self.assertGreater(float(coeff[4]),0)
        rho,report=solve_density_quartic(coeff)
        self.assertAlmostEqual(rho,.3,places=10)
        self.assertLess(abs(report['proposal_surrogate_mse']),1e-15)

    def test_float64_python_coefficients_are_not_first_rounded_to_float32(self):
        c=[.015241578750190521,-.246913578,1.,0.,0.]
        rho,_=solve_density_quartic(c)
        self.assertAlmostEqual(rho,.123456789,places=15)


# ----- test_shadow_source_residual_model -----
def cfg():
    return NS(new_xyz_step_extent=.001,new_rotation_step=.003,new_logscale_step=.02,
              new_opacity_step=.02,new_sh_step=.006,rotation_updates=True)


def shadow_model_fixture(dtype=torch.float64):
    raw=_empty_raw(3,1,'cpu',dtype)
    raw['xyz'][:]=torch.tensor([[.1,-.1,3.],[-.2,.2,3.3],[.2,.1,3.7]],dtype=dtype)
    raw['logscale'][:]=torch.tensor([[.2,.1,.01],[.15,.09,.008],[.12,.07,.01]],dtype=dtype).log()
    raw['opacity'][:]=torch.tensor([[.2837],[1.073],[.83]],dtype=dtype)
    raw['sh_dc'][:]=torch.tensor([[[-0.,.2,-.3]],[[.1,-.2,.3]],[[.2,.1,.3]]],dtype=dtype)
    raw['sh_rest'][:]=torch.tensor([[[.033,-.024,.013],[.047,.021,-.036],[-.026,.017,.041]]],dtype=dtype)
    specs=[dict(old_row=10+i,raw={k:v[i].clone() for k,v in raw.items()}) for i in range(3)]
    m=ShadowSourceResidualModel(1).install_reincarnations(specs)
    m.spatial_lr_scale=1.
    return m.install_details_from_coarse([0,2],detail_alpha=.1),raw


def rendered_proxy(m):
    raw=m.reincarnation_phenotype();alpha=raw['opacity'].sigmoid()
    return (alpha*(raw['sh_dc'].flatten(1).sum(-1,keepdim=True)+raw['xyz'].sum(-1,keepdim=True))).sum()


class ShadowPrefixTests(unittest.TestCase):
    def test_inactive_whole_families_are_transparent_but_native_mother_leaves_survive(self):
        m,source=shadow_model_fixture();raw=model_raw(m)
        self.assertFalse(bool(m.shadow_entity_mask().any()))
        self.assertTrue(bool((raw['opacity'].sigmoid()==0).all()))
        self.assertTrue(same(m._opacity[m.reincarnation_rows],source['opacity']))
        self.assertTrue(bool((m._opacity[m.detail_rows].sigmoid()==0).all()))
        self.assertTrue(all(x['detail_birth_alpha']==0 for x in m.reincarnation_specs))

    def test_unit_probe_opens_only_one_pair_and_never_mutates_latent_or_prefix(self):
        m,source=shadow_model_fixture();latent=m._opacity.detach().clone()
        m.set_shadow_probe(0,1.)
        rows=m.reincarnation_groups[0];mask=torch.zeros(len(m._xyz),dtype=torch.bool);mask[rows]=True
        self.assertTrue(same(m.shadow_entity_mask(),mask))
        raw=model_raw(m)
        self.assertTrue(same(raw['opacity'][rows[0]],source['opacity'][0]))
        torch.testing.assert_close(raw['opacity'][rows[1:]].sigmoid(),torch.full((2,1),.99,dtype=m._xyz.dtype))
        self.assertTrue(bool((raw['opacity'][~mask].sigmoid()==0).all()))
        self.assertTrue(same(m._opacity,latent))
        self.assertFalse(bool(m.shadow_accepted.any()))
        self.assertTrue(all(x['detail_birth_alpha']==0 for x in m.reincarnation_specs))
        m.set_shadow_probe(0,0.)
        self.assertFalse(bool(m.shadow_entity_mask().any()))
        self.assertTrue(bool((model_raw(m)['opacity'].sigmoid()==0).all()))

    def test_accepted_prefix_is_stable_while_the_next_family_is_probed_and_restored(self):
        m,_=shadow_model_fixture();m.accept_shadow_family(0,.03456789123456)
        prefix=model_raw(m);snapshot=m.shadow_state();rows=m.reincarnation_groups[0]
        m.set_shadow_probe(2,.7)
        during=model_raw(m)
        for name in RAW_FIELDS:self.assertTrue(same(during[name][rows],prefix[name][rows]))
        m.install_shadow_state(snapshot)
        restored=model_raw(m)
        for name in RAW_FIELDS:self.assertTrue(same(restored[name],prefix[name]))
        self.assertIsNone(m.shadow_probe_root)
        self.assertEqual(m.reincarnation_specs[0]['detail_birth_alpha'],.99*.03456789123456)

    def test_probe_and_accept_opacity_are_bitexact_with_same_data_rho(self):
        m,_=shadow_model_fixture(torch.float32);rho=.012345678901234
        m.set_shadow_probe(2,rho);before=model_raw(m)
        m.accept_shadow_family(2,rho);after=model_raw(m)
        for name in RAW_FIELDS:self.assertTrue(same(before[name],after[name]),name)
        self.assertEqual(m.reincarnation_specs[2]['detail_birth_alpha'],.99*rho)
        self.assertEqual(float(m.shadow_rho[2]),rho)
        restored=ShadowSourceResidualModel(1).restore(m.capture(),args())
        for name in RAW_FIELDS:self.assertTrue(same(model_raw(restored)[name],after[name]),name)
        self.assertEqual(restored.reincarnation_specs[2]['detail_birth_alpha'],.99*rho)

    def test_rejection_restores_no_entity_without_corrupting_another_accepted_family(self):
        m,source=shadow_model_fixture();m.accept_shadow_family(2,.02)
        accepted=model_raw(m);other=m.reincarnation_groups[2]
        m.set_shadow_probe(0,.8);m.reject_shadow_family(0)
        raw=model_raw(m)
        for name in RAW_FIELDS:self.assertTrue(same(raw[name][other],accepted[name][other]))
        self.assertTrue(bool((raw['opacity'][m.reincarnation_groups[0]].sigmoid()==0).all()))
        self.assertTrue(same(m._opacity[m.reincarnation_rows],source['opacity']))


class ShadowTrainingAndStateTests(unittest.TestCase):
    def test_data_floor_not_fixed_point_one_and_only_accepted_rows_have_gradients(self):
        m,source=shadow_model_fixture();m.accept_shadow_family(0,.02);m.training_setup(args())
        objects=[id(getattr(m,a)) for a in RAW_ATTRIBUTES]
        rendered_proxy(m).backward()
        active=m.shadow_entity_mask();details=m.reincarnation_groups[0][1:]
        self.assertGreater(float(m._features_dc.grad[details].abs().sum()),0.)
        self.assertTrue(bool((m._features_dc.grad[~active]==0).all()))
        self.assertTrue(bool((m._xyz.grad[~active]==0).all()))
        m.optimizer.step();m.optimizer.zero_grad(set_to_none=True);m.project_parameters()
        actual=model_raw(m)['opacity'][details].sigmoid()
        self.assertTrue(bool((actual>=.0198-1.e-15).all()))
        self.assertTrue(bool((actual<.1).all()))
        self.assertTrue(same(model_raw(m)['opacity'][m.reincarnation_rows[0]],source['opacity'][0]))
        self.assertEqual(objects,[id(getattr(m,a)) for a in RAW_ATTRIBUTES])
        m.set_training_stage(2)
        m.reincarnation_phenotype()['opacity'][details].sum().backward()
        m.optimizer.step();m.optimizer.zero_grad(set_to_none=True);m.project_parameters()
        self.assertTrue(bool((model_raw(m)['opacity'][details].sigmoid()<.0198).all()))

    def test_nontrivial_subset_maps_acceptance_probe_and_data_floor_before_rendering(self):
        m,_=shadow_model_fixture();m.accept_shadow_family(2,.0321987654321);m.set_shadow_probe(0,.6)
        rows=m.reincarnation_groups[2];selected=m.reincarnation_subset(rows)
        self.assertEqual(selected.shadow_accepted.tolist(),[True])
        self.assertEqual(float(selected.shadow_rho[0]),.0321987654321)
        self.assertIsNone(selected.shadow_probe_root)
        self.assertTrue(bool(selected.shadow_entity_mask().all()))
        self.assertTrue(bool((model_raw(selected)['opacity'].sigmoid()>0).all()))
        packet=selected.capture();self.assertIn('shadow_source_residual',packet)
        restored=ShadowSourceResidualModel(1).restore(packet,args())
        for name in RAW_FIELDS:self.assertTrue(same(model_raw(restored)[name],model_raw(selected)[name]))
        self.assertEqual(restored._root_ids().tolist(),[12])
        self.assertEqual(restored.reincarnation_specs[0]['detail_birth_alpha'],.99*.0321987654321)
        probe=m.reincarnation_subset(m.reincarnation_groups[0])
        self.assertEqual(probe.shadow_probe_root,0)
        self.assertTrue(bool(probe.shadow_entity_mask().all()))
        self.assertFalse(bool(probe.shadow_accepted.any()))

    def test_empty_subset_and_quantized_hidden_mother_cannot_delete_incumbent(self):
        m,source=shadow_model_fixture();quant=m.reincarnation_quantized_raw(cfg(),1.,source,torch.zeros(len(m._xyz),dtype=torch.long))
        self.assertTrue(bool((quant['opacity'].sigmoid()==0).all()))
        empty=m.reincarnation_subset(torch.empty(0,dtype=torch.long))
        self.assertEqual(len(empty._xyz),0)
        self.assertFalse(bool(empty.shadow_entity_mask().any()))

    def test_nonlattice_predictor_retirement_keeps_accepted_physical_DC_and_true_ATTR_exact(self):
        m,source=shadow_model_fixture(torch.float32);m.accept_shadow_family(0,.03123456789)
        m._features_dc.data[m.reincarnation_groups[0][1:]] += .157731
        m.project_parameters()
        reference={k:v.clone() for k,v in source.items()}
        reference['sh_dc'][0].fill_(.123456)
        reference['sh_dc'][1].fill_(.341927)
        surviving={k:v[1:].clone() for k,v in reference.items()}
        before=None
        for base in (reference,surviving):
            view=QuantizedNewGaussianView(m,cfg(),1.,base,torch.zeros(len(m._xyz),dtype=torch.long))
            raw={k:view.raw_value(k).detach().clone() for k in RAW_FIELDS}
            if before is None:before=raw
            else:
                for name in RAW_FIELDS:self.assertTrue(same(before[name],raw[name]),name)
            appearance={k:raw[k] for k in ATTRIBUTES}
            packet=encode_birth_symbols(appearance,len(base['xyz']),
                symbols={k:view.symbols[k] for k in ATTRIBUTES},steps=view.steps,
                predictor=view.predictor,source_rows=view.source)
            self.assertIsNone(packet)
            packet=encode_birth_attributes(appearance,len(base['xyz']))
            decoded=decode_birth_attributes(packet,base,raw['xyz'],raw['rotation'],birth_sources=view.source)
            for name in RAW_FIELDS:self.assertTrue(same(decoded[name],raw[name]),name)

    def test_invalid_density_pair_and_post_accept_topology_are_rejected(self):
        m,_=shadow_model_fixture()
        for rho in (-.1,1.1,float('nan')):
            with self.assertRaises(ValueError):m.set_shadow_probe(0,rho)
        with self.assertRaises(ValueError):m.set_shadow_probe(1,.2)
        with self.assertRaises(ValueError):m.accept_shadow_family(0,0.)
        m.accept_shadow_family(0,.2)
        with self.assertRaises(ValueError):m.set_shadow_probe(0,.3)
        with self.assertRaises(ValueError):m.install_details_from_coarse([1])


# ----- test_quick_source_controller -----
T=importlib.import_module('pipeline.trainer')


def quick_controller_fixture():
 m,_=dual_fixture();m.config.birth.radiance_mode='quick_source_residual';m.config.rigid_tree.birth_steps=100
 m.birth_branches=None;m.dual_coarse_branches=None;m.dual_training_totals=None
 with torch.no_grad():
  m.gaussians._features_dc[2]=.123456;m.gaussians._features_dc[3]=.341927
  m.gaussians._features_rest[2]=torch.tensor([[.003456,-.012345,.008912],[-.001291,.004341,.009821],[.006101,.003212,-.003731]])
  m.gaussians._features_rest[3]=.219371
 base=model_raw(m.gaussians)
 specs=[dict(old_row=2,raw={k:v[2].clone() for k,v in base.items()},responsibility=1.,split_eligible=True)]
 empty={k:v[:0] for k,v in base.items()}
 model=QuickSourceResidualModel(1).install_reincarnations(specs,novel_raw=empty).install_details_from_coarse([0],detail_alpha=.1)
 views={f'cam{i:02d}.png':NS(path=f'cam{i:02d}.png',camera_center=torch.tensor([i*.1,0.,0.]),image=torch.ones(3,8,8)*.55) for i in range(1,21)}
 m.transaction_camera_views=lambda:views
 names=tuple(views)
 template=NS(group_ids=torch.empty(0,dtype=torch.long),scores=torch.empty(0),support_domains=[],fit_names=names,validation_names=names)
 tx=rb._reincarnation_transaction(m,model,specs,template,None,{});tx.predictive_model=True
 M.Module._install_birth_transaction(m,tx)
 baseline=rb.encode_trial(m,torch.zeros(len(m.births._xyz),dtype=torch.bool),torch.zeros(len(m.gaussians._xyz),dtype=torch.bool))
 m.birth_transaction.old_render_reference=baseline.raw
 m.renderer=SpatialImageRenderer(m.gaussians);m.background=torch.zeros(3)
 m.motion_steps=200;m.total_steps=300;m.birth_cohort=0;m.birth_cohort_start=200;m.birth_end_step=300
 m.motion_committed=True;m.frame_finalized=False;m.trainer=NS(global_step=200)
 m.quick_birth_phase=1;m.quick_birth_optimizer_steps=0;m.quick_birth_initialized=True
 m.birth_control.clear();m.birth_control['quick_schedule']={}
 for name in ('_quick_optimizer_update_done','_commit_population_trial','_commit_opacity_and_retire'):
  setattr(m,name,lambda *args,_name=name,**kwargs:getattr(M.Module,_name)(m,*args,**kwargs))
 m.motion.record_final_diagnostics=lambda:None
 return m,views


class QuickControllerTests(unittest.TestCase):
 def test_exactly100_single_optimizer_updates_protect_then_compete(self):
  m,_=quick_controller_fixture();optimizer=m.births.optimizer
  m._finish_quick_births=Mock();m.finalize_frame=Mock()
  count=len(m.births._xyz);scene_count=len(m.gaussians._xyz)-1+count
  with patch.object(optimizer,'step',wraps=optimizer.step) as calls:
   for step in range(201,301):
    m.trainer.global_step=step
    loss=m.births.get_features.sum()*.001+m.births.get_opacity.sum()*.001
    loss.backward()
    point=torch.zeros(scene_count,3,requires_grad=True);point.grad=torch.ones_like(point)
    package=dict(visibility_filter=torch.ones(scene_count,dtype=torch.bool),radii=torch.ones(scene_count),
      viewspace_points=point,motion_contribution_filter=torch.ones(scene_count,dtype=torch.bool),birth_camera_index=(step-201)%20)
    M.Module.post_training_step(m,package)
    self.assertIs(m.births.optimizer,optimizer)
    if step==249:self.assertEqual(m.births.predictive_phase,1)
    if step==250:self.assertEqual(m.births.predictive_phase,2)
   self.assertEqual(calls.call_count,100)
  self.assertEqual(m.quick_birth_optimizer_steps,100);self.assertEqual(m.birth_end_step,300)
  self.assertIsNone(m.birth_branches);self.assertIsNone(m.birth_transaction.crossfit_state)
  order=[M.Module.transaction_training_view(m,201+i//3,i%3,3)[0].path for i in range(300)]
  self.assertEqual(set(order),set(m.birth_transaction.fit_names))
  self.assertEqual({order.count(name) for name in set(order)},{15})
  m._finish_quick_births.assert_called_once();m.finalize_frame.assert_called_once()

 def test_actual_deleted_parent_heterogeneous_colour_grid_replays_all6_exact(self):
  m,_=quick_controller_fixture();n=len(m.births._xyz)
  with torch.no_grad():m.births._features_dc[m.births.detail_rows]=.589731
  full=torch.ones(n,dtype=torch.bool);deleted=torch.zeros(len(m.gaussians._xyz),dtype=torch.bool);deleted[2]=True
  trial=rb.encode_trial(m,full,deleted)
  # Original source2 is removed; its identical-pose neighbour3 has a different
  # predictor lattice .341927, not the parent's .123456 grid.
  self.assertTrue(bool((trial.keep[trial.sources]==3).all()))
  tx=rb.selected_population(m,trial)
  m._quick_finalizing=True;M.Module._install_birth_transaction(m,tx);m._quick_finalizing=False
  replay=rb.encode_trial(m,torch.ones(n,dtype=torch.bool),tx.provisional_deleted)
  for name in RAW_FIELDS:torch.testing.assert_close(replay.raw[name],trial.raw[name],atol=0,rtol=0)
  self.assertEqual(replay.complete_bytes,trial.complete_bytes)
  self.assertIsNone(m.births.optimizer) # no post-selection/refit optimizer.

 def test_one_all20_initializer_no_sgd_and_refuses_repeat(self):
  m,views=quick_controller_fixture();m.quick_birth_initialized=False
  original_step=m.births.optimizer.step
  def init(module,received,validate_scene):
   self.assertEqual(len(received),20)
   before=validate_scene();after=validate_scene()
   self.assertEqual(before['per_view_mse'],after['per_view_mse'])
   return dict(SGD_steps=0,optimizer_steps=0)
  with (patch('pipeline.model.child_dc_initializer.initialize_child_dc_once',side_effect=init) as init_call,
        patch.object(m.births.optimizer,'step',side_effect=AssertionError('initializer performed SGD'))):
   M.Module._initialize_quick_births(m,views)
   self.assertTrue(m.quick_birth_initialized);self.assertEqual(init_call.call_count,1)
   with self.assertRaises(ValueError):M.Module._initialize_quick_births(m,views)

 def test_hard_total_budget_refuses301_and_silent_dynamic_extension(self):
  T.Trainer._assert_frame_step_budget(NS(current_frame=1,num_steps=300,global_step=300))
  for total,step in ((301,200),(300,301),(2000,0)):
   with self.assertRaises(ValueError):T.Trainer._assert_frame_step_budget(NS(current_frame=1,num_steps=total,global_step=step))

 def test_single_trained_pool_actual_codec_admit_commit_has_no_refit(self):
  m,views=quick_controller_fixture();m.trainer.global_step=300;m.quick_birth_optimizer_steps=100
  m.births.set_training_stage(2);m.quick_birth_phase=2
  with torch.no_grad():m.births._features_dc[m.births.detail_rows]=.589731
  mask=torch.ones(len(m.births._xyz),dtype=torch.bool)
  deleted=torch.zeros(len(m.gaussians._xyz),dtype=torch.bool);deleted[2]=True
  target=rb.encode_trial(m,mask,deleted)
  previous=m.renderer.gaussians;m.renderer.gaussians=rb.RawView(target.raw,1,m.gaussians)
  for view in views.values():view.image=m.renderer(view,m.background)['render'].detach().clone()
  m.renderer.gaussians=previous
  m.birth_transaction.observations.fill_((1<<20)-1)
  del m.birth_branches # fresh midframe resume never instantiates legacy dual fields.
  with patch.object(m.births.optimizer,'step',side_effect=AssertionError('post-admission SGD')):
   M.Module._finish_quick_births(m)
  proof=m.birth_control['quick_committed']
  self.assertEqual(proof['candidate_count'],3)
  self.assertEqual(proof['actual_training_steps'],dict(motion=200,birth_iterations=100,birth_optimizer_updates=100,total=300))
  self.assertEqual(proof['post_admission_SGD_steps'],0)
  self.assertTrue(proof['replay_checks'][0]['exact_codec_replay'])
  self.assertIsNone(getattr(m,'birth_branches',None));self.assertIsNone(m.births.optimizer)


# ----- test_shadow_source_controller -----
def shadow_controller_fixture():
 m,views=quick_controller_fixture();m.config.birth.radiance_mode='shadow_source_residual'
 base=model_raw(m.gaussians)
 specs=[dict(old_row=row,raw={k:v[row].clone() for k,v in base.items()},
             responsibility=3-index,split_eligible=True) for index,row in enumerate((2,8,18))]
 empty={k:v[:0] for k,v in base.items()}
 model=ShadowSourceResidualModel(1).install_reincarnations(specs,novel_raw=empty).install_details_from_coarse([0,1,2])
 model.set_training_stage(1)
 names=tuple(views)
 template=NS(group_ids=torch.empty(0,dtype=torch.long),scores=torch.empty(0),
             support_domains=[],fit_names=names,validation_names=names)
 tx=rb._reincarnation_transaction(m,model,specs,template,None,{});tx.predictive_model=True
 M.Module._install_birth_transaction(m,tx)
 baseline=rb.encode_trial(m,torch.zeros(len(m.births._xyz),dtype=torch.bool),torch.zeros(len(m.gaussians._xyz),dtype=torch.bool))
 m.birth_transaction.old_render_reference=baseline.raw
 m.quick_birth_initialized=False;m.shadow_initialized_state=None
 for name in ('_capture_birth_pool','_restore_birth_pool','_finish_shadow_births'):
  setattr(m,name,lambda *args,_name=name,**kwargs:getattr(M.Module,_name)(m,*args,**kwargs))
 m.finalize_frame=Mock()
 return m,views,baseline


def trial_for(m,roots,rho=.04):
 mask=torch.zeros(len(m.births._xyz),dtype=torch.bool)
 deleted=torch.zeros(len(m.gaussians._xyz),dtype=torch.bool)
 for root in roots:
  rows=m.births.reincarnation_groups[root];mask[rows]=True
  deleted[m.birth_transaction.replacements[rows[0]]]=True
 return rb.encode_trial(m,mask,deleted)


def image_of(m,raw,view):
 old=m.renderer.gaussians
 try:
  m.renderer.gaussians=rb.RawView(raw,1,m.gaussians)
  return m.renderer(view,m.background)['render'].detach().clone()
 finally:m.renderer.gaussians=old


class ShadowControllerTests(unittest.TestCase):
 def test_reject_first_accept_second_probe_third_exact_prefix_and_sgd_replay(self):
  m,views,baseline=shadow_controller_fixture();model=m.births
  with torch.no_grad():model._features_dc[model.reincarnation_groups[1][1:]]=.6
  model.set_shadow_probe(1,.04);target=trial_for(m,[1]);model.clear_shadow_probe()
  for view in views.values():view.image=image_of(m,target.raw,view)
  first=next(iter(views.values()));calls=[]
  def propose(module,received,root,render_density,probe_scene):
   self.assertEqual(len(received),20);calls.append(root)
   before=render_density(first,0.)
   reference=baseline.raw if root<2 else target.raw
   torch.testing.assert_close(before,image_of(m,reference,first),atol=0,rtol=0)
   raw,children=probe_scene()
   # Packed scene excludes rejected family0; family1 survives while family2
   # is transient. Full-pool child indices are never used as packed indices.
   self.assertEqual(len(raw['xyz']),len(m.gaussians._xyz)+(2 if root<2 else 4))
   torch.testing.assert_close(children,torch.tensor([len(raw['xyz'])-2,len(raw['xyz'])-1]))
   if root==2:
    torch.testing.assert_close(raw['xyz'][-6],target.raw['xyz'][-3],atol=0,rtol=0)
    module.births._features_dc[module.births.reincarnation_groups[root][1:]]=.43
   return dict(rho=.04 if root==1 else 0.,SGD_steps=0,optimizer_steps=0)
  original_step=model.optimizer.step
  with patch('pipeline.model.shadow_birth_initializer.propose_shadow_family',side_effect=propose),\
       patch.object(model.optimizer,'step',side_effect=AssertionError('shadow initializer performed SGD')):
   M.Module._initialize_shadow_births(m,views)
  self.assertEqual(calls,[0,1,2]);self.assertTrue(m.quick_birth_initialized)
  self.assertEqual(len(m.births._xyz),3);self.assertEqual(m.births.reincarnation_specs[0]['old_row'],8)
  self.assertAlmostEqual(m.births.reincarnation_specs[0]['detail_birth_alpha'],.99*.04)
  self.assertLess(m.births.reincarnation_specs[0]['detail_birth_alpha'],.1)
  self.assertIsNotNone(m.birth_transaction.fixed_reference);self.assertEqual(m.birth_end_step,300)
  self.assertEqual(len(m.births.optimizer.state),0);m.finalize_frame.assert_not_called()
  replay=rb.CommitTrial.restore(m.shadow_initialized_state['trial'],'cpu')
  for name in RAW_FIELDS:torch.testing.assert_close(replay.raw[name],target.raw[name],atol=0,rtol=0)
  snapshot=M.Module._capture_birth_pool(m)
  M.Module._restore_birth_pool(m,snapshot)
  self.assertEqual(len(m.births.reincarnation_groups),1)
  self.assertAlmostEqual(float(m.births.shadow_rho[0]),.04)

 def test_all_rejected_finishes200_without_empty_optimizer_updates(self):
  m,views,baseline=shadow_controller_fixture();m._finish_shadow_births=Mock()
  with patch('pipeline.model.shadow_birth_initializer.propose_shadow_family',return_value=dict(rho=0.)):
   M.Module._initialize_shadow_births(m,views)
  self.assertEqual(len(m.births._xyz),0);self.assertEqual(m.birth_end_step,200)
  self.assertEqual(m.quick_birth_optimizer_steps,0);m._finish_shadow_births.assert_called_once()
  m.finalize_frame.assert_called_once()
  self.assertEqual(m.birth_control['quick_schedule']['birth_steps'],0)
  replay=rb.CommitTrial.restore(m.shadow_initialized_state['trial'],'cpu')
  for name in RAW_FIELDS:torch.testing.assert_close(replay.raw[name],baseline.raw[name],atol=0,rtol=0)

 def test_empty_population_runs_real200_final_admission_with_zero_sgd(self):
  m,views,_=shadow_controller_fixture()
  def finish():
   # SimpleNamespace fixture emulates Module.total_steps' dynamic property.
   m.total_steps=m.birth_end_step
   return M.Module._finish_shadow_births(m)
  m._finish_shadow_births=finish
  with patch('pipeline.model.shadow_birth_initializer.propose_shadow_family',return_value=dict(rho=0.)):
   M.Module._initialize_shadow_births(m,views)
  proof=m.birth_control['quick_committed']
  self.assertEqual(proof['candidate_count'],0)
  self.assertEqual(proof['actual_training_steps'],dict(motion=200,birth_iterations=0,birth_optimizer_updates=0,total=200))
  self.assertEqual(proof['post_admission_SGD_steps'],0)
  m.finalize_frame.assert_called_once()

 def test_final_atomic_rejection_restores_one_mother_and_preserves_other_family(self):
  m,views,_=shadow_controller_fixture();model=m.births
  model.accept_shadow_family(1,.04);model.accept_shadow_family(2,.03)
  target=trial_for(m,[2]);first=next(iter(views.values()))
  for view in views.values():view.image=image_of(m,target.raw,view)
  model.set_training_stage(2)
  m.birth_transaction.observations.fill_((1<<20)-1)
  chosen,report=rb.admit(m,m.config.birth.residual,list(views.values()),list(views.values()),
                       capture_prefixes=True,atomic_lineage=True)
  self.assertTrue(bool((chosen.keep==8).any()))
  self.assertFalse(bool((chosen.keep==18).any()))
  self.assertEqual(len(chosen.candidates),3)
  self.assertEqual(set(chosen.candidates.tolist()),set(model.reincarnation_groups[2].tolist()))
  for name in RAW_FIELDS:
   old=target.raw[name][:len(target.keep)][torch.searchsorted(target.keep,chosen.keep)]
   expected=torch.cat((old,target.raw[name][len(target.keep):]),0)
   torch.testing.assert_close(chosen.raw[name],expected,atol=0,rtol=0)
  self.assertEqual(report['candidate_groups_evaluated'],2)

 def test_training_damage_rolls_back_same_accepted_initialization_without_sgd(self):
  m,views,_=shadow_controller_fixture();model=m.births
  model.set_shadow_probe(1,.04);target=trial_for(m,[1]);model.clear_shadow_probe()
  for view in views.values():view.image=image_of(m,target.raw,view)
  def proposal(module,received,root,render_density,probe_scene):
   return dict(rho=.04 if root==1 else 0.)
  with patch('pipeline.model.shadow_birth_initializer.propose_shadow_family',side_effect=proposal):
   M.Module._initialize_shadow_births(m,views)
  m.trainer.global_step=300;m.quick_birth_optimizer_steps=100
  m.births.set_training_stage(2);m.quick_birth_phase=2
  with torch.no_grad():m.births._features_dc[m.births.detail_rows]=-1.
  m.birth_transaction.observations.fill_((1<<20)-1)
  with patch.object(m.births.optimizer,'step',side_effect=AssertionError('rollback did SGD')):
   M.Module._finish_shadow_births(m)
  proof=m.birth_control['quick_committed']
  self.assertEqual(proof['selected_trajectory_state'],'accepted_initialization')
  self.assertEqual(proof['actual_training_steps']['total'],300)
  self.assertEqual(proof['candidate_count'],3)
  replay,_=materialize_quantized_gaussians(model_raw(m.gaussians),m.make_birth_view())
  for name in RAW_FIELDS:torch.testing.assert_close(replay[name],target.raw[name],atol=0,rtol=0)


# ----- test_quick_source_residual_model -----
def quick_model_fixture():
    raw=_empty_raw(3,1,'cpu',torch.float64)
    raw['xyz'][:]=torch.tensor([[.1,-.1,3.],[-.2,.2,3.3],[.2,.1,3.7]],dtype=torch.float64)
    raw['logscale'][:]=torch.tensor([[.2,.1,.01],[.15,.09,.008],[.12,.07,.01]],dtype=torch.float64).log()
    raw['opacity'][:]=torch.tensor([[.2837],[1.073],[.83]],dtype=torch.float64)
    raw['sh_dc'][:]=torch.tensor([[[-0.,.2,-.3]],[[.1,-.2,.3]],[[.2,.1,.3]]],dtype=torch.float64)
    raw['sh_rest'][:]=torch.tensor([[[.033,-.024,.013],[.047,.021,-.036],[-.026,.017,.041]]],dtype=torch.float64)
    specs=[dict(old_row=i,raw={k:v[i].clone() for k,v in raw.items()}) for i in range(3)]
    m=QuickSourceResidualModel(1).install_reincarnations(specs)
    m.spatial_lr_scale=1.
    return m.install_details_from_coarse([0,2]),raw


def quick_objective(m):
    return sum(value.sum() for value in m.reincarnation_phenotype().values())


class QuickDCModelTests(unittest.TestCase):
    def test_single_P_pool_has_no_common_or_angular_training(self):
        m,raw=quick_model_fixture();m.training_setup(args())
        self.assertEqual(m.novel_count,0)
        self.assertFalse(m._features_rest.requires_grad)
        self.assertFalse(m.p_common_dc.requires_grad)
        self.assertFalse(m.p_common_sh1.requires_grad)
        quick_objective(m).backward()
        self.assertGreater(float(m._features_dc.grad[m.detail_rows].abs().sum()),0.)
        self.assertIsNone(m._features_rest.grad)
        self.assertIsNone(m.p_common_dc.grad)
        m.optimizer.step();m.optimizer.zero_grad(set_to_none=True);m.project_parameters()
        self.assertTrue(same(m.p_common_dc,raw['sh_dc']))
        self.assertTrue(same(m.p_common_sh1,raw['sh_rest']))
        self.assertTrue(same(model_raw(m)['sh_rest'],raw['sh_rest'][m._point_root]))
        self.assertNotIn(m._features_rest,m.optimizer.state)

    def test_DC_change_with_unrelated_codec_predictor_never_rounds_or_projects_SHrest(self):
        m,source=quick_model_fixture()
        predictor={k:v.clone()+.317 for k,v in source.items()}
        before=m.reincarnation_quantized_raw(config(),1.,predictor,torch.zeros(len(m._xyz),dtype=torch.long))
        m._features_dc.data[m.detail_rows] += .157
        after=m.reincarnation_quantized_raw(config(),1.,predictor,torch.zeros(len(m._xyz),dtype=torch.long))
        self.assertTrue(same(before['sh_rest'],after['sh_rest']))
        self.assertTrue(same(after['sh_rest'],source['sh_rest'][m._point_root]))
        self.assertFalse(same(before['sh_dc'][m.detail_rows],after['sh_dc'][m.detail_rows]))
        self.assertTrue(same(after['sh_dc'][m.reincarnation_rows],source['sh_dc']))

    def test_fixed_rest_DC_box_keeps_source_feasible_and_no_unbounded_HDR(self):
        m,_=quick_model_fixture();lower,upper=m.intrinsic_dc_bounds();prior,_=m._prior_features()
        self.assertTrue(bool((lower<=prior).all() & (upper>=prior).all()))
        m._features_dc.data[m.detail_rows] += 1000.
        m.project_parameters();raw=model_raw(m)
        self.assertTrue(bool((raw['sh_dc']>=lower).all() & (raw['sh_dc']<=upper).all()))
        lo,hi=sh1_sphere_range(raw['sh_dc'],raw['sh_rest']);boundlo,boundhi=m._radiance_bounds()
        self.assertTrue(bool((lo>=boundlo-2.e-14).all() & (hi<=boundhi+2.e-14).all()))

    def test_density_protection_then_parent_child_competition_in_same_optimizer(self):
        m,source=quick_model_fixture();m.training_setup(args());optimizer=m.optimizer
        rows,floor=m._maturity_opacity_floor()
        m.reincarnation_phenotype()['opacity'].sum().backward()
        m.optimizer.step();m.optimizer.zero_grad(set_to_none=True);m.project_parameters()
        self.assertTrue(bool((model_raw(m)['opacity'][rows]>=floor).all()))
        self.assertTrue(same(model_raw(m)['opacity'][m.reincarnation_rows],source['opacity']))
        m.set_training_stage(2)
        m.reincarnation_phenotype()['opacity'].sum().backward()
        self.assertGreater(float(m._opacity.grad[m.reincarnation_rows].abs().sum()),0.)
        m.optimizer.step();m.optimizer.zero_grad(set_to_none=True);m.project_parameters()
        self.assertIs(m.optimizer,optimizer)
        self.assertTrue(bool((model_raw(m)['opacity'][rows]<floor).all()))

    def test_internal_compatibility_endpoints_are_exact_two_vertex_degeneracy(self):
        m,_=quick_model_fixture();m._features_dc.data[m.detail_rows] += .14;m.project_parameters()
        m.freeze_predictive_endpoints()
        for left,right in [('prior','common'),('detail','full')]:
            for name in RAW_FIELDS:self.assertTrue(same(m.endpoint_raw(left)[name],m.endpoint_raw(right)[name]))
        m.apply_predictive_weights([0.,1.,0.]);a=model_raw(m)
        m.apply_predictive_weights([0.,0.,1.]);b=model_raw(m)
        for name in RAW_FIELDS:self.assertTrue(same(a[name],b[name]))

    def test_capture_subset_and_restore_keep_one_model_flags_and_physical_SH(self):
        m,source=quick_model_fixture();m.training_setup(args());quick_objective(m).backward()
        m.optimizer.step();m.optimizer.zero_grad(set_to_none=True);m.project_parameters()
        m.set_training_stage(2)
        ids=m.reincarnation_groups[2]
        selected=m.reincarnation_subset(ids)
        packet=selected.capture();self.assertIn('quick_source_residual',packet)
        restored=QuickSourceResidualModel(1).restore(packet,args())
        self.assertFalse(restored._features_rest.requires_grad)
        self.assertTrue(same(model_raw(restored)['sh_rest'],source['sh_rest'][2:3].expand(3,3,3)))
        for name in RAW_FIELDS:self.assertTrue(same(model_raw(selected)[name],model_raw(restored)[name]))
        self.assertTrue(restored.reincarnation_state()['quick_source_residual'])
        self.assertNotIn('branches',restored.reincarnation_state())

    def test_P_only_rejects_novel_rows_and_higher_SH_is_not_expanded(self):
        m=QuickSourceResidualModel(1)
        with self.assertRaisesRegex(ValueError,'P-only'):
            m.install_reincarnations([],novel_raw=_empty_raw(1,1,'cpu',torch.float64))
        with self.assertRaises(NotImplementedError):QuickSourceResidualModel(3)


# ----- test_early_replacement_model -----
def early_args():
    return NS(position_lr_init=.001,rotation_lr=.001,scaling_lr=.005,
              opacity_lr=.02,feature_lr=.01)


def early_model_fixture(dtype=torch.float64):
    raw=_empty_raw(7,1,'cpu',dtype)
    raw['xyz'][:]=torch.arange(21,dtype=dtype).reshape(7,3)*.031
    raw['rotation'][:]=torch.tensor([.93,.12,-.19,.27],dtype=dtype)
    raw['logscale'][:]=torch.tensor([-.4,-.9,-1.3],dtype=dtype)
    raw['opacity'][:]=torch.linspace(-.7,1.3,7,dtype=dtype)[:,None]
    raw['sh_dc'][:]=torch.linspace(-.2,.6,21,dtype=dtype).reshape(7,1,3)
    raw['sh_rest'][:]=torch.linspace(-.07,.09,63,dtype=dtype).reshape(7,3,3)
    rows=torch.tensor([4,1])
    seed={k:v[rows].clone() for k,v in raw.items()}
    model=EarlyReplacementModel(1).install(seed,rows)
    return model,raw


def optimizer(model):
    motion=nn.Parameter(torch.tensor([.13],dtype=model.local_xyz.dtype))
    shared=torch.optim.Adam([dict(params=[motion],name='motion',lr=.003)],eps=1e-8)
    model.attach_optimizer(shared,early_args())
    return shared,motion


def early_objective(model):
    # Every learner group is exercised, including local rotation at identity.
    raw=model.raw()
    return sum(v.sum() for k,v in raw.items() if k!='opacity')+model.get_opacity.sum()


class EarlyReplacementTests(unittest.TestCase):
    def test_alpha_maturity_freeze_capture_and_release_has_own_adam_age(self):
        m,_=early_model_fixture();shared,_=optimizer(m);m.set_alpha_trainable(False)
        for _ in range(2):
            shared.zero_grad(set_to_none=True);early_objective(m).backward();shared.step()
        self.assertNotIn(m._opacity,shared.state)
        clone=m.subset([1]);self.assertFalse(clone._opacity.requires_grad)
        clone.set_alpha_trainable(True)
        fresh,_=optimizer(clone)
        fresh.zero_grad(set_to_none=True);early_objective(clone).backward();fresh.step()
        self.assertEqual(int(fresh.state[clone._opacity]['step']),1)
        self.assertEqual(int(fresh.state[clone.local_xyz]['step']),3)

    def test_initial_physical_identity_predictor_independence_and_covariance(self):
        m,old=early_model_fixture()
        before=model_raw(m)
        for name in RAW_FIELDS:self.assertTrue(same(before[name],old[name][m.parent_rows]),name)
        view=QuantizedNewGaussianView(m,config(),1.,old,torch.tensor([0,6]))
        changed={k:v+(.341927 if k!='rotation' else .1) for k,v in old.items()}
        other=QuantizedNewGaussianView(m,config(),1.,changed,torch.tensor([5,3]))
        for name in RAW_FIELDS:
            self.assertTrue(same(view.raw_value(name),before[name]),name)
            self.assertTrue(same(other.raw_value(name),before[name]),name)
        self.assertTrue(torch.equal(m.get_covariance(),view.get_covariance()))

    def test_probability_alpha_endpoints_and_pristine_real_quantview_gradient(self):
        m,old=early_model_fixture(torch.float32)
        seed={k:v[m.parent_rows].clone() for k,v in old.items()}
        seed['opacity'][:]=torch.tensor([[100.],[.713]])
        alpha=seed['opacity'].sigmoid();alpha[1]=torch.nextafter(alpha[1],torch.ones_like(alpha[1]))
        m.install(seed,m.parent_rows,alpha)
        shared,_=optimizer(m)
        for canonical in (False,True):
            m.bind_codec_alpha(canonical)
            view=QuantizedNewGaussianView(m,config(),1.,old,torch.tensor([0,6]))
            expected=seed['opacity'].sigmoid() if canonical else alpha
            self.assertTrue(same(view.get_opacity,expected))
            shared.zero_grad(set_to_none=True)
            view.get_opacity.sum().backward()
            self.assertTrue(torch.equal(m._opacity.grad,torch.ones_like(alpha)))
        shared.step();m.project_parameters()
        self.assertLess(float(m._opacity[0]),1.)
        view=QuantizedNewGaussianView(m,config(),1.,old,torch.tensor([0,6]))
        self.assertTrue(same(view.get_opacity,view.raw_value('opacity').sigmoid()))
        # Raw-hook users also retain the interior canonical-logit derivative.
        shared.zero_grad(set_to_none=True)
        view.raw_value('opacity').sum().backward()
        self.assertTrue(bool((m._opacity.grad>0).all()))

    def test_twenty_fresh_view_graphs_keep_live_motion_and_all_six_gradients(self):
        m,old=early_model_fixture()
        translation=nn.Parameter(torch.full_like(old['xyz'],.01))
        rotation=nn.Parameter(torch.full_like(old['rotation'],.02))
        shared,_=optimizer(m)
        shared.add_param_group(dict(params=[translation,rotation],lr=.001,name='live_pose'))
        for view_index in range(20):
            m.bind_source_pose(old['xyz']+translation*(view_index+1)/20,
                               old['rotation']+rotation*(view_index+1)/20)
            view=QuantizedNewGaussianView(m,config(),1.,old,torch.tensor([0,6]))
            loss=(view.get_xyz.sum()+view.get_rotation.sum()+view.get_scaling.sum()+
                  view.get_opacity.sum()+view.get_features.sum())
            loss.backward()
        for _,attr in m.GROUPS:
            self.assertTrue(bool(torch.isfinite(getattr(m,attr).grad).all()),attr)
            self.assertGreater(float(getattr(m,attr).grad.abs().sum()),0.,attr)
        selected=torch.zeros(7,dtype=torch.bool);selected[m.parent_rows]=True
        self.assertTrue(bool((translation.grad[~selected]==0).all()))
        self.assertTrue(bool((rotation.grad[~selected]==0).all()))
        self.assertGreater(float(translation.grad[selected].abs().sum()),0.)
        self.assertGreater(float(rotation.grad[selected].abs().sum()),0.)
        shared.step()

    def test_detached_motion_after_200_keeps_local_coordinates_and_shared_adam_age(self):
        m,old=early_model_fixture();shared,motion=optimizer(m)
        for _ in range(130):
            shared.zero_grad(set_to_none=True)
            (early_objective(m)+motion.sum()).backward();shared.step();m.project_parameters()
        params={a:getattr(m,a) for _,a in m.GROUPS}
        moments={a:shared.state[p]['exp_avg'].clone() for a,p in params.items()}
        translated=old['xyz'].clone().requires_grad_()
        m.bind_source_pose(translated,old['rotation'],detach=True)
        for a,p in params.items():
            self.assertIs(getattr(m,a),p)
            self.assertEqual(int(shared.state[p]['step']),130)
            self.assertTrue(torch.equal(shared.state[p]['exp_avg'],moments[a]))
        shared.zero_grad(set_to_none=True);early_objective(m).backward()
        self.assertIsNone(translated.grad)
        shared.step()
        for _,a in m.GROUPS:self.assertEqual(int(shared.state[getattr(m,a)]['step']),131)

    def test_nontrivial_subset_slices_shared_moments_and_restore_continues_exactly(self):
        m,old=early_model_fixture();shared,motion=optimizer(m)
        for _ in range(3):
            shared.zero_grad(set_to_none=True);early_objective(m).backward();shared.step();m.project_parameters()
        before={a:shared.state[getattr(m,a)]['exp_avg'][1:2].clone() for _,a in m.GROUPS}
        motion_group=shared.param_groups[0]
        subset=m.subset([1])
        self.assertIsNone(subset.optimizer)
        self.assertEqual(m.parent_rows.tolist(),[4,1])
        m.select_rows_inplace([1],shared)
        self.assertIs(m.optimizer,shared);self.assertIs(shared.param_groups[0],motion_group)
        self.assertEqual(m.parent_rows.tolist(),[1])
        for name,a in m.GROUPS:
            parameter=getattr(m,a);state=shared.state[parameter]
            self.assertEqual(int(state['step']),3)
            self.assertTrue(torch.equal(state['exp_avg'],before[a]))
            group=next(g for g in shared.param_groups if g.get('name')==name)
            self.assertIs(group['params'][0],parameter)
            self.assertEqual(group['eps'],1e-15)
        clone=EarlyReplacementModel(1)
        newmotion=nn.Parameter(motion.detach().clone())
        newshared=torch.optim.Adam([dict(params=[newmotion],lr=.003,name='motion')])
        clone.restore(m.capture(),newshared,early_args())
        for instance,opt in ((m,shared),(clone,newshared)):
            opt.zero_grad(set_to_none=True);early_objective(instance).backward();opt.step();instance.project_parameters()
        for name in RAW_FIELDS:self.assertTrue(same(model_raw(m)[name],model_raw(clone)[name]),name)
        self.assertEqual(int(newshared.state[clone.local_xyz]['step']),4)
        empty=m.subset([]);self.assertEqual(len(empty.parent_rows),0)

    def test_source_envelope_is_immutable_and_all_sphere_bounded(self):
        m,old=early_model_fixture()
        m._features_dc.data.fill_(20.);m._features_rest.data.fill_(-30.)
        m.project_parameters()
        low,high=sh1_sphere_range(m.raw()['sh_dc'],m.raw()['sh_rest'])
        self.assertTrue(bool((low>=m._low-1e-12).all()))
        self.assertTrue(bool((high<=m._high+1e-12).all()))
        q=old['rotation'].clone();q[:,1:]*=2
        m.bind_source_pose(old['xyz']+1,q)
        self.assertTrue(torch.equal(m._seed_raw['sh_dc'],old['sh_dc'][m.parent_rows]))


# ----- test_early_replacement_controller -----
class EvidenceRenderer(SpatialImageRenderer):
 """CPU differentiable adapter; codecs are real, native proof is separate."""
 def __call__(self,view,bg,override_color=None):
  g=self.gaussians;xyz=g.get_xyz;scale=g.get_scaling
  y,x=torch.meshgrid(torch.linspace(-.15,.65,8),torch.linspace(-.15,.65,8),indexing='ij')
  d=((x[None]-xyz[:,0,None,None])/scale[:,0,None,None]).square()+((y[None]-xyz[:,1,None,None])/scale[:,1,None,None]).square()
  alpha=g.get_opacity[:,0,None,None]*torch.exp(-.5*d)
  trans=torch.cumprod(torch.cat((torch.ones_like(alpha[:1]),1-alpha),0),0)[:-1]
  colour=.5+.28209479177387814*g.get_features[:,0] if override_color is None else override_color
  render=torch.einsum('nhw,nc->chw',alpha*trans,colour)
  n=len(xyz);mask=(alpha*trans).flatten(1).sum(1)>0
  return dict(render=render,radii=torch.ones(n),visibility_filter=torch.ones(n,dtype=torch.bool),motion_contribution_filter=mask)


def early_controller_fixture(directory):
 m,_=quick_controller_fixture();m.config.birth.radiance_mode='early_source_replacement'
 m.config.birth.residual.repair_candidates=2;m.config.birth.residual.max_candidates=4
 m.config.rigid_tree.birth_steps=100;m.motion.config.birth_steps=100
 m.config.rigid_tree.rate_loss_weight=0.
 m.motion_steps=200;m.total_steps=300;m.motion_committed=False;m.frame_finalized=False;m.frame_prepared=True
 m.opacity_update.committed=False;m.opacity_update.delta.requires_grad_(True)
 m.opacity_update.config.learning_rate=.0001
 with torch.no_grad():
  m.opacity_update.delta.zero_();m.opacity_update.delta[4]=.05;m.opacity_update.delta[2]=.02
  m.gaussians._opacity.copy_(m.opacity_update.canonical_logits())
 for v in m.motion.parameters():v.requires_grad_(True)
 for attr in ('_xyz','_rotation','_scaling','_opacity','_features_dc','_features_rest'):getattr(m.gaussians,attr).requires_grad_(False)
 m.early_initialized=False;m.early_birth_updates=0;m.early_shared_steps=0
 m.early_selected=None;m.early_pinned_rows=None;m.early_pinned_values=None;m.early_focus_weights={}
 m.output_dir=Path(directory);m.trainer=NS(global_step=0);m.finalize_frame=Mock()
 m.birth_control.clear();m.birth_cohort=0;m.birth_cohort_start=70;m.birth_end_step=300
 views={f'cam{i:02d}.png':NS(path=f'cam{i:02d}.png',camera_center=torch.tensor([i*.1,0.,0.]),image=torch.full((3,8,8),.55)) for i in range(1,21)}
 m.transaction_camera_views=lambda:views;m.renderer=EvidenceRenderer(m.gaussians);m.background=torch.zeros(3)
 for name,value in vars(EarlyReplacementController).items():
  if name.startswith('_early_') and callable(value):setattr(m,name,MethodType(value,m))
 m._rate_byte_price=MethodType(M.Module._rate_byte_price,m)
 m.forward=MethodType(M.Module.forward,m)
 m.transaction_training_view=MethodType(M.Module.transaction_training_view,m)
 m.post_training_step=MethodType(M.Module.post_training_step,m)
 m._early_attach_opacity();m.motion.record_final_diagnostics=lambda:None
 return m,views


class EarlyControllerTests(unittest.TestCase):
 def test_one_shared300_with_pinned_parent_actual200_subset_and_whole_rollback(self):
  with tempfile.TemporaryDirectory() as directory:
   m,views=early_controller_fixture(directory);optimizer=m.motion.optimizer
   n=len(m.gaussians._xyz);selected_rows=torch.tensor([4,2]);observed=torch.full((n,),(1<<20)-1,dtype=torch.long)
   scores=torch.zeros(n);scores[selected_rows]=torch.tensor([3.,2.])
   responsibility=dict(old_RGB_SSE=10.,score_semantics='fixture responsibility; native has its own proof')
   saved={};accepted200={};pinned_values=None;unselected70=None;seen=[];end_steps={}
   finish=m._early_finish
   def witness_finish():
    end_steps.update({name:int(optimizer.state[getattr(m.births,attr)]['step']) for name,attr in m.births.GROUPS})
    return finish()
   m._early_finish=witness_finish
   def training(view,step,regularization_weight=1):
    package=m.forward(view,training=True)
    loss=(package['render']-view.image).square().mean()
    if m.early_initialized:
     target=.9 if m.motion_committed else 0.
     loss=loss+(m.births.get_features-target).square().mean()+m.births.get_opacity.sum()*.001
     loss=loss+(m.births.get_xyz.square().mean()+m.births.get_scaling.square().mean()+m.births.get_rotation.sum())*.0001
    seen.append((step,Path(view.path).name))
    return loss,dict(image_loss=loss.detach(),psnr=loss.detach()*0,gs=n),package
   m.training_step=training
   trainer=NS(module=m,config=NS(motion_views_per_step=2,birth_views_per_step=3))
   trainer._views_per_step=MethodType(T.Trainer._views_per_step,trainer)
   fetcher=NS(next=lambda:views['cam01.png'])
   def choose(module,config,*args,**kwargs):
    # Isolate orchestration from greedy ranking. All packet/source/decode paths
    # remain real, and keep nontrivial local row1/original parent2.
    mask=torch.tensor([False,True]);delete=torch.zeros(n,dtype=torch.bool);delete[2]=True
    trial=rb.encode_trial(module,mask,delete)
    saved.update({name:{k:v.clone() for k,v in optimizer.state[getattr(module.births,attr)].items()} for name,attr in module.births.GROUPS if getattr(module.births,attr) in optimizer.state})
    previous=module.renderer.gaussians;module.renderer.gaussians=rb.RawView(trial.raw,1,module.gaussians)
    for view in views.values():view.image=module.renderer(view,module.background)['render'].detach().clone()
    module.renderer.gaussians=previous
    accepted200.update(raw={k:v.clone() for k,v in trial.raw.items()},bytes=trial.complete_bytes)
    return trial,dict(complete_byte_budget=trial.complete_bytes+100000)
   with (patch('pipeline.early_replacement.score_support_responsibility',return_value=(scores,observed,observed,responsibility)),
         patch('pipeline.early_replacement.admit',side_effect=choose),
         patch.object(optimizer,'step',wraps=optimizer.step) as calls,
         patch.object(m.opacity_update.optimizer,'step',side_effect=AssertionError('second OPD Adam step'))):
    for step in range(1,301):
     m.trainer.global_step=step;m._early_prepare_rates(step)
     # Actual trainer view routing/backward and Module.forward/post-step paths.
     T.Trainer._fit_views(trainer,fetcher,step,defer_metrics=True)
     self.assertIs(m.motion.optimizer,optimizer)
     if m.early_initialized:self.assertIs(m.births.optimizer,optimizer)
     if step==70:
      self.assertEqual(m.births.parent_rows.tolist(),[4,2]);pinned_values=m.opacity_update.delta[m.early_pinned_rows].clone()
      unselected70=m.opacity_update.delta[3].clone()
      self.assertFalse(m.births._opacity.requires_grad)
     if 70<step<200:
      torch.testing.assert_close(m.opacity_update.delta[m.early_pinned_rows],pinned_values,atol=0,rtol=0)
      torch.testing.assert_close(m.opacity_update.alphas()[selected_rows],m.births._seed_alpha,atol=0,rtol=0)
      state=optimizer.state[m.opacity_update.delta]
      for key in ('exp_avg','exp_avg_sq'):self.assertEqual(float(state[key][m.early_pinned_rows].abs().sum()),0.)
      torch.testing.assert_close(m.births.get_opacity,m.births._seed_alpha,atol=0,rtol=0)
      self.assertNotIn(m.births._opacity,optimizer.state)
     if step==199:self.assertFalse(torch.equal(m.opacity_update.delta[3],unselected70))
     if step==200:
      self.assertEqual(m.births.parent_rows.tolist(),[2]);self.assertTrue(m.births._opacity.requires_grad)
      self.assertNotIn(m.births._opacity,optimizer.state)
      for name,attr in m.births.GROUPS:
       if name=='early_opacity':continue
       state=optimizer.state[getattr(m.births,attr)]
       self.assertEqual(int(state['step']),130)
       for key in ('exp_avg','exp_avg_sq'):torch.testing.assert_close(state[key],saved[name][key][[1]],atol=0,rtol=0)
      replay=rb.encode_trial(m,torch.ones(1,dtype=torch.bool),m.birth_transaction.provisional_deleted)
      for name in RAW_FIELDS:torch.testing.assert_close(replay.raw[name],accepted200['raw'][name],atol=0,rtol=0)
      # The phase3 forward population is the actual oldkeep+new codec order,
      # not merely a permutation-equivalent original-slot training scene.
      with torch.no_grad():
       actual=m.forward(views['cam01.png'],training=False)['render']
       previous=m.renderer.gaussians;m.renderer.gaussians=rb.RawView(accepted200['raw'],1,m.gaussians)
       expected=m.renderer(views['cam01.png'],m.background)['render'];m.renderer.gaussians=previous
       torch.testing.assert_close(actual,expected,atol=0,rtol=0)
     if step==299:
      self.assertEqual(int(optimizer.state[m.births._features_dc]['step']),229)
      self.assertEqual(int(optimizer.state[m.births._opacity]['step']),99)
    self.assertEqual(calls.call_count,300)
   proof=m.birth_control['early_replacement']
   self.assertEqual(proof['global_iterations'],300);self.assertEqual(proof['motion_iterations'],200)
   self.assertEqual(proof['newborn_iterations'],230);self.assertEqual(proof['joint_overlap_iterations'],130)
   self.assertEqual(proof['final_population_iterations'],100);self.assertEqual(proof['extra_SGD_steps'],0)
   self.assertEqual(end_steps,{name:(100 if name=='early_opacity' else 230) for name,_ in m.births.GROUPS})
   self.assertEqual(len(seen),300*2)
   for step in range(1,301):self.assertEqual(sum(s==step for s,_ in seen),2)
   for start in range(71,301,10):
    self.assertCountEqual([name for s,name in seen if start<=s<start+10],views)
   self.assertFalse(proof['accepted_final100']);m.finalize_frame.assert_called_once()
   self.assertEqual(int(optimizer.state[m.births._features_dc]['step']),130) # full200 rollback
   self.assertNotIn(m.births._opacity,optimizer.state)
   self.assertEqual(m.birth_control['committed_trial_bytes'],accepted200['bytes'])


# ----- test_persistent_newborn_model -----
def persistent_args():return NS(percent_dense=.01,feature_lr=.01,opacity_lr=.02)


def spec(uid='tri:a',born=1,dtype=torch.float64,offset=0.):
    return dict(uid=uid,born_frame=born,track_ids=[1,2,3],raw=dict(
        xyz=torch.tensor([.1+offset,-.1,3.],dtype=dtype),
        rotation=torch.tensor([.923456,.123456,-.234567,.034567],dtype=dtype),
        logscale=torch.tensor([.03,.02,.004],dtype=dtype).log(),
        opacity=torch.tensor([-2.1972245773362196],dtype=dtype),
        sh_dc=torch.tensor([[-0.,.123456,-.234567]],dtype=dtype),
        sh_rest=torch.tensor([[.03,-.02,.01],[.04,.02,-.03],[-.02,.01,.04]],dtype=dtype)))


def update(m,count=1):
    for _ in range(count):
        raw=m.raw()
        loss=(raw['sh_dc']-.4).square().sum()+(raw['opacity'].sigmoid()-.6).square().sum()
        loss.backward();m.optimizer.step();m.optimizer.zero_grad(set_to_none=True);m.project_parameters()


def step(m,uid,name='f_dc'):
    return int(m.optimizer.state.get(m.uid_parameters(uid)[name],{}).get('step',0))


class PersistentAgeTests(unittest.TestCase):
    def test_100_then_200_old_uid_and_new_uid_starts_zero_in_one_adam(self):
        m=PersistentNewbornModel(1).install([spec()],old_rows=[-1],frame_index=1)
        m.training_setup(persistent_args());optimizer=m.optimizer;dc=m.uid_parameters('tri:a')['f_dc']
        update(m,100);self.assertEqual(step(m,'tri:a'),100)
        state=m.bank_state()
        m.install([spec(),spec('tri:b',2,offset=.1)],previous_state=state,old_rows=[40,-1],frame_index=2)
        self.assertIs(m.optimizer,optimizer);self.assertIs(m.uid_parameters('tri:a')['f_dc'],dc)
        self.assertEqual(step(m,'tri:a'),100);self.assertEqual(step(m,'tri:b'),0)
        fresh=PersistentNewbornModel(1).install([spec('tri:b',2,offset=.1)],frame_index=2)
        fresh.training_setup(persistent_args())
        update(m,100);update(fresh,100)
        self.assertEqual(step(m,'tri:a'),200);self.assertEqual(step(m,'tri:b'),100)
        for name in ('f_dc','opacity'):
            torch.testing.assert_close(m.uid_parameters('tri:b')[name],fresh.uid_parameters('tri:b')[name],rtol=0,atol=0)
        self.assertEqual(m.fresh_mask.tolist(),[False,True])
        self.assertEqual(m.born_frames.tolist(),[1,2])
        self.assertEqual(m.age_frames.tolist(),[2,1])

    def test_maturity_floor_uses_own_age_and_is_not_restarted_next_frame(self):
        m=PersistentNewbornModel(1).install([spec()],frame_index=1);m.training_setup(persistent_args())
        floor=spec()['raw']['opacity']
        for _ in range(49):
            m.raw()['opacity'].sum().backward();m.optimizer.step();m.optimizer.zero_grad(set_to_none=True);m.project_parameters()
        self.assertTrue(bool((m.raw()['opacity'][0]>=floor).all()))
        m.raw()['opacity'].sum().backward();m.optimizer.step();m.optimizer.zero_grad(set_to_none=True);m.project_parameters()
        self.assertEqual(step(m,'tri:a','opacity'),50)
        self.assertTrue(bool((m.raw()['opacity'][0]<floor).all()))
        opacity=m.uid_parameters('tri:a')['opacity'].detach().clone()
        m.install([spec()],frame_index=2,old_rows=[18])
        self.assertEqual(step(m,'tri:a','opacity'),50)
        self.assertTrue(same(m.raw()['opacity'][0],opacity))

    def test_rollback_restores_uid_values_and_exact_moments_and_drops_fresh_reject(self):
        m=PersistentNewbornModel(1).install([spec()],frame_index=1);m.training_setup(persistent_args());update(m,100)
        previous=m.bank_state();raw_before=model_raw(m);leaf=m.uid_parameters('tri:a')['f_dc']
        m.install([spec(),spec('tri:b',2)],previous_state=previous,frame_index=2,old_rows=[31,-1]);update(m,7)
        m.rollback_uids(['tri:a','tri:b'],previous)
        self.assertEqual(m.uids,('tri:a',));self.assertIs(m.uid_parameters('tri:a')['f_dc'],leaf)
        self.assertEqual(step(m,'tri:a'),100)
        for key in RAW_FIELDS:self.assertTrue(same(model_raw(m)[key],raw_before[key]),key)
        for name,p in m.uid_parameters('tri:a').items():
            for key,value in previous['optimizer']['tri:a'][name].items():
                if torch.is_tensor(value):self.assertTrue(same(m.optimizer.state[p][key],value))


class PersistentPhysicalTests(unittest.TestCase):
    def test_geometry_and_SH_fixed_but_DC_alpha_receive_real_graph_gradients(self):
        s=spec();m=PersistentNewbornModel(1).install([s]);m.training_setup(persistent_args())
        before=model_raw(m);update(m,2);after=model_raw(m)
        for name in ('xyz','rotation','logscale','sh_rest'):
            self.assertTrue(same(before[name],after[name]),name)
            self.assertFalse(m.birth_parameter_value(name).requires_grad)
        self.assertFalse(same(before['sh_dc'],after['sh_dc']))
        self.assertFalse(same(before['opacity'],after['opacity']))
        self.assertEqual([g['name'] for g in m.optimizer.param_groups],['f_dc','opacity'])
        self.assertEqual(sum(len(g['params']) for g in m.optimizer.param_groups),2)

    def test_rebind_committed_six_raw_keeps_nonunit_q_bits_and_Adam_age(self):
        m=PersistentNewbornModel(1).install([spec()]);m.training_setup(persistent_args());update(m,100)
        before=model_raw(m);moment=m.bank_state()['optimizer']
        changed={k:v.clone() for k,v in before.items()};changed['xyz']+=.01;changed['opacity']+=.02
        m.rebind(m.uids,changed,old_rows=[77],frame_index=2)
        for name in RAW_FIELDS:self.assertTrue(same(m.raw()[name],changed[name]),name)
        self.assertEqual(step(m,'tri:a'),100);self.assertEqual(m.old_rows.tolist(),[77])
        for name,p in m.uid_parameters('tri:a').items():
            self.assertTrue(same(m.optimizer.state[p]['exp_avg'],moment['tri:a'][name]['exp_avg']))
        bad={k:v.clone() for k,v in changed.items()};bad['sh_rest']+=.001
        with self.assertRaisesRegex(ValueError,'fixed physical SH1'):m.rebind(m.uids,bad)

    def test_physical_raw_does_not_depend_on_predictor_and_ATTR_pose_is_exact(self):
        s=spec(dtype=torch.float32);m=PersistentNewbornModel(1).install([s]);m.training_setup(persistent_args());update(m,2)
        baseline=model_raw(m)
        for phase in (.123456,.341927):
            reference={k:v.repeat((2,)+(1,)*(v.ndim-1)) for k,v in baseline.items()}
            reference['sh_dc'].fill_(phase)
            view=QuantizedNewGaussianView(m,config(),1.,reference,torch.zeros(1,dtype=torch.long))
            raw={k:view.raw_value(k).detach() for k in RAW_FIELDS}
            for name in RAW_FIELDS:self.assertTrue(same(raw[name],baseline[name]),name)
            materialized,_=materialize_quantized_gaussians(reference,view)
            self.assertTrue(same(materialized['rotation'][-1:],baseline['rotation']))
            appearance={k:raw[k] for k in ATTRIBUTES}
            packet=encode_birth_symbols(appearance,2,symbols={k:view.symbols[k] for k in ATTRIBUTES},
                steps=view.steps,predictor=view.predictor,source_rows=view.source)
            self.assertIsNone(packet)
            decoded=decode_birth_attributes(encode_birth_attributes(appearance,2),reference,
                raw['xyz'],raw['rotation'],birth_sources=view.source)
            for name in RAW_FIELDS:self.assertTrue(same(decoded[name],raw[name]),name)


class PersistentBankTests(unittest.TestCase):
    def test_subset_capture_restore_remaps_UID_and_Adam_without_reset_or_alias(self):
        m=PersistentNewbornModel(1).install([spec(),spec('tri:b',2,offset=.2)],old_rows=[41,88],frame_index=2)
        m.training_setup(persistent_args());update(m,100)
        selected=m.subset(['tri:b']);packet=selected.capture()
        restored=PersistentNewbornModel(1).restore(packet,persistent_args())
        self.assertEqual(restored.uids,('tri:b',));self.assertEqual(restored.old_rows.tolist(),[88])
        self.assertEqual(step(restored,'tri:b'),100)
        for name in RAW_FIELDS:self.assertTrue(same(model_raw(restored)[name],model_raw(m)[name][1:]))
        restored.uid_parameters('tri:b')['f_dc'].data.add_(.02)
        self.assertFalse(same(restored.raw()['sh_dc'],selected.raw()['sh_dc']))
        self.assertEqual(len(m.subset([]).uids),0)

    def test_full_bank_over_512_merges_active_subset_without_GPU_inactive_parameters(self):
        seed=PersistentNewbornModel(1).install([spec()]);seed.training_setup(persistent_args());update(seed,100)
        previous=seed.bank_state();record=previous['records']['tri:a'];optim=previous['optimizer']['tri:a']
        ids=[f'tri:{i}' for i in range(513)]
        previous['uids']=ids;previous['records']={};previous['optimizer']={}
        for i,u in enumerate(ids):
            previous['records'][u]=copy.deepcopy(record);previous['records'][u]['old_row']=1000+i
            previous['records'][u]['raw']['xyz'][0]+=i*.001
            previous['optimizer'][u]=copy.deepcopy(optim)
        active=PersistentNewbornModel(1).install([],uids=[ids[2]],previous_state=previous,frame_index=2)
        active.training_setup(persistent_args());update(active,1)
        self.assertEqual(len(active.uids),1);self.assertEqual(step(active,ids[2]),101)
        raws={k:torch.stack([previous['records'][u]['raw'][k] for u in ids]) for k in RAW_FIELDS}
        for key in RAW_FIELDS:raws[key][2]=active.raw()[key][0].detach()
        raws['xyz']+=.002
        merged=PersistentNewbornModel.merge_bank_states(previous,active.bank_state(),ids,raws,
            list(range(9000,9513)),frame_index=2)
        self.assertEqual(len(merged['uids']),513)
        self.assertEqual(int(merged['optimizer'][ids[2]]['f_dc']['step']),101)
        self.assertEqual(int(merged['optimizer'][ids[512]]['f_dc']['step']),100)
        for name in ('f_dc','opacity'):
            self.assertTrue(same(merged['optimizer'][ids[512]][name]['exp_avg'],previous['optimizer'][ids[512]][name]['exp_avg']))
        self.assertEqual(merged['records'][ids[512]]['old_row'],9512)
        restored=PersistentNewbornModel(1).install([],uids=[ids[512]],previous_state=merged)
        restored.training_setup(persistent_args());self.assertEqual(step(restored,ids[512]),100)
        self.assertTrue(same(restored.raw()['xyz'][0],raws['xyz'][512]))

    def test_original_UIDs_invalid_layout_and_higher_SH_are_explicit(self):
        s=spec();s['born_frame']=0
        with self.assertRaisesRegex(ValueError,'dynamic-frame'):PersistentNewbornModel(1).install([s])
        with self.assertRaisesRegex(ValueError,'unique'):PersistentNewbornModel(1).install([spec(),spec()])
        with self.assertRaisesRegex(ValueError,'same old row'):
            PersistentNewbornModel(1).install([spec(),spec('tri:b')],old_rows=[7,7])
        with self.assertRaises(NotImplementedError):PersistentNewbornModel(3)


# ----- test_support_reincarnation_proposal -----
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

    def test_complete_raw_copy_no_scale_cap_and_real_error_mass_report(self):
        module,views,raw,_=support_fixture();before=model_raw(module.gaussians)
        specs,report=propose_support_reincarnations(module,views,source_quota=2,prefilter=3)
        self.assertEqual([s['old_row'] for s in specs],[0,3])
        self.assertGreater(float(specs[0]['raw']['logscale'].exp().max()),1.)
        self.assertEqual(report['prefilter_rows'],[0,3,2])
        self.assertAlmostEqual(report['selected_fraction_of_total_SSE'],.5,places=6)
        self.assertIsNone(report['source_scale_limit'])
        self.assertFalse(report['signed_DC_gradient_used'])
        for spec in specs:
            for name in RAW_FIELDS:
                torch.testing.assert_close(spec['raw'][name],raw[name][spec['old_row']],rtol=0,atol=0)
            self.assertEqual(spec['raw']['sh_rest'].shape,(15,3))
            self.assertEqual(spec['support_world'].shape,(9,3))
            self.assertIn('not_new_surface',spec['support_kind'])
        for name,value in model_raw(module.gaussians).items():torch.testing.assert_close(value,before[name],rtol=0,atol=0)
        specs[0]['raw']['xyz'].add_(10)
        torch.testing.assert_close(module.gaussians._xyz,before['xyz'],rtol=0,atol=0)

    def test_constant_error_is_selected_for_coarse_without_fake_detail_pattern(self):
        module,views,_,_=support_fixture(constant=True)
        specs,report=propose_support_reincarnations(module,views,2,3)
        self.assertEqual(len(specs),2)
        self.assertGreater(report['selected_native_responsibility_SSE'],0)
        self.assertTrue(all(not s['split_eligible'] for s in specs))

    def test_fine_direction_uses_original_axis_order_and_observable_pattern(self):
        module,views,_,_=support_fixture()
        # Identity rotation makes the source's local tangent pattern explicit.
        with torch.no_grad():module.gaussians._rotation[:]=torch.tensor([1.,0.,0.,0.])
        specs,_=propose_support_reincarnations(module,views,3,3)
        source=next(s for s in specs if s['old_row']==3)
        torch.testing.assert_close(source['support_axes'],torch.tensor([0,1]))
        self.assertTrue(source['split_eligible'])
        self.assertGreater(source['contrast_evidence'],0)
        self.assertGreater(abs(float(source['tangent_offset'][0])),abs(float(source['tangent_offset'][1])))
        self.assertGreater(float(source['support_mass'].sum()),0)
        self.assertIn('not per-probe Talpha',source['support_semantics'])

    def test_all_zero_error_has_no_births_and_no_visibility_guess(self):
        module,views,_,_=support_fixture()
        for v in views:v.rendered=v.image.clone()
        specs,report=propose_support_reincarnations(module,views,2,3)
        self.assertEqual(specs,[])
        self.assertEqual(report['excluded_exact_zero'],4)
        self.assertEqual(report['positive_responsibility_count'],0)
        self.assertEqual(report['observed_old_count'],3)

    def test_corner_only_direction_outside_family_domain_keeps_coarse_explicitly(self):
        from pipeline.model.support_reincarnation_proposal import _split_direction
        local=torch.tensor([[[x,y] for y in (-1.,0.,1.) for x in (-1.,0.,1.)]])
        count=torch.zeros(1,9);count[0,0]=3;count[0,-1]=3
        mass=torch.zeros(1,9);mass[0,0]=.5;mass[0,-1]=.5
        pattern=torch.zeros(1,9,3);pattern[0,0]=-.2;pattern[0,-1]=.2
        probes=dict(local=local,counts=count,mass=mass,pattern=pattern,
            coherent=pattern.square(),arithmetic_bound=torch.zeros(1))
        eligible,offset,evidence,reason,rho2=_split_direction(probes,0)
        self.assertFalse(eligible)
        self.assertEqual(reason,'outside_family_one_sigma_domain')
        self.assertAlmostEqual(rho2,2.)
        self.assertGreater(evidence,0)
        torch.testing.assert_close(offset.abs(),torch.ones(2),atol=0,rtol=0)

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


class ComplementaryCameraTests(unittest.TestCase):
    def circle(self,count):
        views=[]
        for i in range(count):
            angle=2*math.pi*i/count;center=torch.tensor([2*math.cos(angle),2*math.sin(angle),.3],dtype=torch.float64)
            forward=torch.nn.functional.normalize(-center,dim=0)
            right=torch.nn.functional.normalize(torch.cross(torch.tensor([0.,0.,1.],dtype=torch.float64),forward,dim=0),dim=0)
            up=torch.cross(forward,right,dim=0)
            inverse=torch.eye(4,dtype=torch.float64);inverse[:3,:3]=torch.stack((right,up,forward));inverse[3,:3]=center
            views.append(SimpleNamespace(path=f'cam{i+1:02d}.png',world_view_transform=torch.linalg.inv(inverse)))
        return views

    def test_all_twenty_training_views_are_balanced_complements(self):
        views=self.circle(20);groups,report=camera_complementary_groups(views)
        self.assertEqual(tuple(map(len,groups)),(10,10))
        self.assertFalse(set(groups[0]).intersection(groups[1]))
        self.assertEqual(set(groups[0]+groups[1]),{v.path for v in views})
        for pair in report['pairing']:
            self.assertEqual(sum(p in groups[0] for p in pair),1)
            self.assertEqual(sum(p in groups[1] for p in pair),1)
        self.assertLess(report['feature_mean_difference_norm'],.3)
        self.assertEqual(camera_complementary_groups(list(reversed(views)))[0],groups)
        self.assertFalse(report['external_test_used'])

    def test_odd_camera_count_is_partitioned_without_dropping_a_view(self):
        groups,_=camera_complementary_groups(self.circle(5))
        self.assertEqual(sorted(map(len,groups)),[2,3])
        self.assertEqual(len(set(groups[0]+groups[1])),5)
        views=self.circle(4);views[0].path='cam00.png'
        with self.assertRaisesRegex(ValueError,'external test'):camera_complementary_groups(views)


# ----- test_reincarnation_controller -----
def raw(n=2,offset=0.):
 return dict(xyz=torch.tensor([[.1,.2,3.]]).repeat(n,1)+offset,
   rotation=torch.tensor([[1.,0.,0.,0.]]).repeat(n,1),logscale=torch.full((n,3),-2.),
   opacity=torch.zeros(n,1),sh_dc=torch.full((n,1,3),.1234),sh_rest=torch.zeros(n,3,3))
def model():
 m=BirthGaussianModel(1)
 for name,attr in zip(RAW_FIELDS,RAW_ATTRIBUTES):setattr(m,attr,torch.nn.Parameter(raw()[name]))
 m.novel_mask=torch.tensor([True,False]);return m


class CrossfitControllerTests(unittest.TestCase):
 def test_nonfitted_views_have_no_N_gradient_but_identical_forward(self):
  m=model();view=QuantizedNewGaussianView(m,M.BirthConfig(),1.,raw(),torch.tensor([0,1]))
  before={k:view.raw_value(k).detach().clone() for k in RAW_FIELDS}
  view.row_gradient_scale=torch.tensor([0.,1.])
  (view.get_xyz.sum()+view.get_features.sum()).backward()
  self.assertEqual(float(m._xyz.grad[0].abs().sum()),0.)
  self.assertGreater(float(m._xyz.grad[1].abs().sum()),0.)
  self.assertEqual(float(m._features_dc.grad[0].abs().sum()),0.)
  for k in RAW_FIELDS:torch.testing.assert_close(view.raw_value(k),before[k],atol=0,rtol=0)
  for attr in RAW_ATTRIBUTES:getattr(m,attr).grad=None
  view=QuantizedNewGaussianView(m,M.BirthConfig(),1.,raw(),torch.tensor([0,1]))
  view.row_gradient_scale=torch.tensor([2.,1.]);view.locked_geometry_rows=m.novel_mask
  (view.get_xyz.sum()+view.get_features.sum()).backward()
  self.assertEqual(float(m._xyz.grad[0].abs().sum()),0.)
  torch.testing.assert_close(m._features_dc.grad[0],2*m._features_dc.grad[1])

 def test_both_independent_geometries_must_improve_their_unfitted_group(self):
  m=model();m.novel_mask=torch.ones(2,dtype=torch.bool)
  a,b=raw(offset=1.),raw(offset=2.)
  state=dict(camera_groups={'A':('a',),'B':('b',)},novel_A_raw=a,novel_B_raw=b)
  tx=NS(proposed_spatial=2,replacements=torch.tensor([-1,-1]),crossfit_state=state)
  module=NS(births=m,birth_transaction=tx,device=torch.device('cpu'),gaussians=NS(_xyz=torch.zeros(3,3)))
  values={1:(.1,1.1),2:(.5,.1)}
  def encode(module,mask,deleted):
   branch=0 if not bool(mask.any()) else round(float(module.births._xyz[0,0])-.1)
   return NS(raw={'branch':branch},complete_bytes=10)
  def errors(module,views,r):return torch.tensor(values.get(r['branch'],(1.,1.)))
  with patch.object(rb,'encode_trial',side_effect=encode),patch.object(rb,'validation_errors',side_effect=errors):
   result=rb.qualify_crossfit_novel(module,{'a':None,'b':None})
   self.assertFalse(result['eligible']) # global mean gain cannot excuse A failing B.
   self.assertEqual(result['hypotheses']['A']['heldout_cameras'],['b'])
   self.assertEqual(result['hypotheses']['B']['heldout_cameras'],['a'])
   values[1]=(.1,.4)
   result=rb.qualify_crossfit_novel(module,{'a':None,'b':None},state)
   self.assertTrue(result['eligible']);self.assertEqual(result['selected_branch'],'A')
   torch.testing.assert_close(m._xyz,a['xyz'],atol=0,rtol=0)

 def test_second_geometry_starts_from_identical_initial_six_fields(self):
  fitted=model();fitted.novel_mask=torch.ones(2,dtype=torch.bool);initial=raw(offset=4.);snapshot=model_raw(fitted)
  fitted.install_families_from_coarse=lambda roots,tangent_offsets:copy.deepcopy(fitted)
  tx=NS(crossfit_state=dict(novel_initial_raw=initial,training_branch='A'),proposed_spatial=2,
        reincarnation_specs=[],region_diagnostics={})
  module=NS(birth_transaction=tx,births=fitted,device=torch.device('cpu'),gaussians=NS(_xyz=torch.zeros(3,3)))
  seen={}
  def capture(module,nextmodel,specs,template,state,*args,**kwargs):
   seen.update(raw=model_raw(nextmodel),state=state);return NS(sources=torch.arange(2))
  with patch.object(rb,'_reincarnation_transaction',side_effect=capture):
   _,report=rb.refine_reincarnations(module,NS(max_candidates=512),{})
  for key in RAW_FIELDS:
   torch.testing.assert_close(seen['raw'][key],initial[key],atol=0,rtol=0)
   torch.testing.assert_close(seen['state']['novel_A_raw'][key],snapshot[key],atol=0,rtol=0)
   torch.testing.assert_close(model_raw(fitted)[key],snapshot[key],atol=0,rtol=0)
  self.assertEqual(seen['state']['training_branch'],'B')
  self.assertEqual(report['optimizer_reset'],'entire_fixed_cohort')

 def test_interleaved_views_keep_all20_and_each_update_has_both_groups(self):
  groups={'A':tuple(f'a{i}' for i in range(10)),'B':tuple(f'b{i}' for i in range(10))}
  names=tuple(sorted(groups['A']+groups['B']));views={name:name for name in names}
  module=NS(config=NS(birth=NS(mechanism='residual_transaction',radiance_mode='support_reincarnation')),
    birth_terminal=False,motion_steps=200,birth_transaction=NS(fit_names=names,crossfit_state={'camera_groups':groups}),
    transaction_camera_views=lambda:views)
  seen=[]
  for step in range(201,221):
   batch=[M.Module.transaction_training_view(module,step,i,3)[0] for i in range(3)]
   self.assertTrue(any(x.startswith('a') for x in batch));self.assertTrue(any(x.startswith('b') for x in batch));seen+=batch
  self.assertEqual(set(seen),set(names));self.assertEqual(set(seen.count(x) for x in names),{3})

 def test_native_support_trial_selected_reference_and_checkpoint_are_exact(self):
  from test_surface import fixture
  from pipeline.model.reincarnation_model import ReincarnationModel
  from pipeline.model.birth_quantization import materialize_quantized_gaussians
  module,_=fixture();module.config.birth.radiance_mode='support_reincarnation'
  original=model_raw(module.gaussians)
  specs=[dict(old_row=2,raw={key:value[2].clone() for key,value in original.items()},
      responsibility=1.,split_eligible=True,tangent_offset=torch.tensor([.01,.01]))]
  initial=raw(1)
  candidate=ReincarnationModel(1).install_reincarnations(specs,novel_raw=initial)
  views=module.transaction_camera_views();names=tuple(views)
  template=NS(group_ids=torch.tensor([0]),scores=torch.ones(1),support_domains=[{}],fit_names=names,validation_names=names)
  crossfit=dict(camera_groups={'A':names[:1],'B':names[1:]},training_branch='A',
      novel_initial_raw=initial,novel_A_raw=initial,novel_B_raw=initial,eligible=True)
  transaction=rb._reincarnation_transaction(module,candidate,specs,template,crossfit,{})
  M.Module._install_birth_transaction(module,transaction)
  coarse_snapshot=dict(model=module.births.capture(),transaction=transaction.capture())
  refined,_=rb.refine_reincarnations(module,module.config.birth.residual,views)
  M.Module._install_birth_transaction(module,refined)
  fine_view=module.make_birth_view();family=module.births.family_rows
  self.assertEqual(tuple(family.shape),(1,3))
  self.assertTrue(bool((fine_view.source[family]==fine_view.source[family[:,:1]]).all()))
  self.assertEqual(len(fine_view.raw_value('xyz')),4)
  M.Module._restore_birth_pool(module,coarse_snapshot)
  # Even a different surviving source lattice cannot perturb pristine P.
  deleted=torch.zeros(len(module.gaussians._xyz),dtype=torch.bool);deleted[2]=True
  trial=rb.encode_trial(module,torch.tensor([False,True]),deleted)
  for key in RAW_FIELDS:
   torch.testing.assert_close(trial.raw[key][-1],original[key][2],atol=0,rtol=0)
   torch.testing.assert_close(model_raw(module.gaussians)[key],original[key],atol=0,rtol=0)
  transaction=rb.selected_population(module,trial)
  self.assertEqual(transaction.proposed_spatial,0)
  self.assertIsNone(transaction.family_state)
  module.birth_terminal=True
  M.Module._install_birth_transaction(module,transaction)
  full,_=materialize_quantized_gaussians(transaction.fixed_reference,module.make_birth_view())
  for key in RAW_FIELDS:torch.testing.assert_close(full[key],trial.raw[key],atol=0,rtol=0)
  snapshot=dict(model=module.births.capture(),transaction=transaction.capture())
  M.Module._restore_birth_pool(module,snapshot)
  self.assertTrue(module.births._reincarnation_terminal)
  full,_=materialize_quantized_gaussians(transaction.fixed_reference,module.make_birth_view())
  for key in RAW_FIELDS:torch.testing.assert_close(full[key],trial.raw[key],atol=0,rtol=0)


if __name__ == '__main__':
    unittest.main()
