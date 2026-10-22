"""Constant-size birth competition, fixed-pool refit and exact stream resume."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.module import (BirthConfig, GaussianConfig, Module, ModuleConfig,
                             balanced_replacement_rows)
from pipeline.model.birth_quantization import (RAW_ATTRIBUTES, RAW_FIELDS,
    QuantizedNewGaussianView, model_raw)
from pipeline.model.gaussian import GaussianModel, BirthGaussianModel
from pipeline.model.sparse_updates import OpacityUpdateConfig, QuantizedOpacityUpdate
from pipeline.model.spatial_gated_motion import SpatialGatedMotion
from pipeline.trainer import Trainer
from scripts.replay_rigid_tree import decode_frame
from _motion_fixtures import current_motion
from _stream_fixtures import Rows, reference_raw


class BalancedSelectionTests(unittest.TestCase):
    def select(self, old, new, sources):
        return balanced_replacement_rows(torch.tensor(old), torch.tensor(new),
                                         torch.tensor(sources, dtype=torch.long))

    def test_weak_births_lose_to_old_rows_and_equal_alpha_avoids_churn(self):
        rows, births = self.select([.1, .9, .8, .7], [.6, .05], [1, 2])
        self.assertEqual(rows.tolist(), [1, 2, 3])
        self.assertEqual(births.tolist(), [0])
        rows, births = self.select([.5, .5], [.5], [0])
        self.assertEqual(rows.tolist(), [0, 1])
        self.assertEqual(births.tolist(), [])

    def test_zero_births_do_not_independently_delete_low_alpha_old_rows(self):
        rows, births = self.select([.00001, .2, .9], [], [])
        self.assertEqual(rows.tolist(), [0, 1, 2])
        self.assertEqual(births.tolist(), [])

    def test_required_low_alpha_source_survives_and_other_old_row_retires(self):
        rows, births = self.select([.01, .2, .8, .9], [.95], [0])
        self.assertEqual(rows.tolist(), [0, 2, 3])
        self.assertEqual(births.tolist(), [0])

    def test_source_dependencies_and_excess_candidates_keep_exact_population(self):
        for old, new, sources in (
                ([.2, .8], [.9, .7, .6], [0, 1, 1]),
                ([.1], [.9, .8], [0, 0]),
                ([.1, .2, .3, .4], [.99, .98, .97, .96], [0, 1, 2, 3])):
            with self.subTest(old=old):
                rows, births = self.select(old, new, sources)
                self.assertEqual(len(rows) + len(births), len(old))
                self.assertTrue(len(rows))
                self.assertTrue(set(sources[i] for i in births).issubset(rows.tolist()))
                if len(old) == 4:
                    self.assertEqual(len(births), 2)

    def test_competition_uses_rendered_quantized_alpha_instead_of_raw_candidates(self):
        old = {name: value[:2].clone() for name, value in reference_raw().items()}
        old['opacity'] = torch.tensor([[.02], [2.]])
        newborn = {name: value[1:].clone() for name, value in old.items()}
        newborn['opacity'][:] = .049
        source = torch.tensor([1])
        view = QuantizedNewGaussianView(Rows(newborn),
            BirthConfig(new_opacity_step=.1), 2., old, source)
        _, incorrect_births = balanced_replacement_rows(old['opacity'].sigmoid(),
            newborn['opacity'].sigmoid(), source)
        _, correct_births = balanced_replacement_rows(old['opacity'].sigmoid(),
            view.get_opacity, source)
        self.assertEqual(incorrect_births.tolist(), [0])
        self.assertEqual(correct_births.tolist(), [])

    def test_tolerance_protects_sources_with_at_most_ten_percent_net_growth(self):
        old = torch.arange(1, 51).float() / 100
        for candidates, expected_growth in ((9, 0), (10, 1), (20, 2)):
            sources = torch.arange(candidates)
            rows, births = balanced_replacement_rows(old, torch.full((candidates,), .9),
                sources, count_tolerance=.1)
            retired = len(old) - len(rows)
            self.assertEqual(len(births) - retired, expected_growth)
            self.assertLessEqual(abs(len(births) - retired), int(.1 * len(births)))
            self.assertTrue(set(sources[births].tolist()).issubset(rows.tolist()))

    def test_tolerance_does_not_force_growth_or_round_up_small_birth_counts(self):
        rows, births = balanced_replacement_rows(torch.tensor([.1, .9, .8]),
            torch.tensor([.6]), torch.tensor([1]), count_tolerance=.1)
        self.assertEqual(len(rows) + len(births), 3)
        self.assertEqual(len(births), 1)
        for invalid in (True, -.1, 1., float('nan')):
            with self.subTest(value=invalid), self.assertRaises(ValueError):
                BirthConfig(replacement_policy='balanced_current_births',
                    replacement_count_tolerance=invalid).validate(200)
        with self.assertRaises(ValueError):
            BirthConfig(replacement_count_tolerance=.1).validate(200)

    def test_committed_tolerant_replacement_preserves_its_selected_row_counts(self):
        holder = SimpleNamespace(
            config=SimpleNamespace(birth=SimpleNamespace(replacement_policy='balanced_current_births')),
            birth_control=dict(replacement_committed=True, retirement=dict(
                birth_count=10, retired_count=9, previous_count=50)),
            gaussians=SimpleNamespace(_xyz=torch.zeros(41, 3)),
            births=SimpleNamespace(_xyz=torch.zeros(10, 3)))
        Module.commit_birth_replacement(holder)
        holder.births._xyz = torch.zeros(9, 3)
        with self.assertRaisesRegex(ValueError, 'fixed birth row set'):
            Module.commit_birth_replacement(holder)

    def test_invalid_sources_and_alpha_fail_before_mutating_models(self):
        for old, new, sources in (([], [], []), ([.1], [.9], [1]),
                                  ([float('nan')], [], []), ([.2], [1.1], [0])):
            with self.subTest(old=old), self.assertRaises(ValueError):
                self.select(old, new, sources)
        for policy in ('old', True):
            with self.assertRaises(ValueError):
                BirthConfig(replacement_policy=policy).validate(200)
        with self.assertRaises(ValueError):
            BirthConfig(replace_low_opacity=True,
                        replacement_policy='balanced_current_births').validate(200)


class PoseRenderer:
    """A differentiable pose objective exercising the real merged model path."""
    def __init__(self, model):
        self.gaussians = model

    def __call__(self, viewpoint, background):
        return dict(render=self.gaussians.get_xyz)


@unittest.skipUnless(torch.cuda.is_available(), 'Native Gaussian optimizer uses CUDA buffers')
class BalancedLifecycleCudaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    @staticmethod
    def gaussian(raw, config, *, births=False, sources=None):
        model = BirthGaussianModel(1) if births else GaussianModel(1)
        for name, attr in zip(RAW_FIELDS, RAW_ATTRIBUTES):
            setattr(model, attr, torch.nn.Parameter(raw[name].cuda().clone()))
        model.spatial_lr_scale = 1.
        model.active_sh_degree = 1
        model.training_setup(config)
        model.max_radii2D = torch.zeros(len(model._xyz), device='cuda')
        if births:
            model.source_rows = sources.cuda().clone()
        return model

    def active(self, directory):
        motion_cfg = copy.deepcopy(current_motion(points_per_axis=2,
            max_depth=2, amplitude_refit_steps=20).config)
        config = ModuleConfig(rigid_tree=motion_cfg,
            birth=BirthConfig(replacement_policy='balanced_current_births', refit_steps=40,
                              rotation_updates=False),
            opacity_update=OpacityUpdateConfig(enabled=True, mode='alpha', step=.01))
        raw = reference_raw()
        raw['logscale'].fill_(-3.)
        raw['opacity'] = torch.logit(torch.linspace(.1, .8, 8)[:, None])
        data = SimpleNamespace(current_frame=1, cameras_extent=2.)
        module = Module(config, data)
        module.trainer = SimpleNamespace(global_step=260, device=torch.device('cuda'),
            output_dir=Path(directory), config=SimpleNamespace(num_init_steps=15000),
            datamodule=data, module=module)
        module.gaussians = self.gaussian(raw, config.gaussian)
        module.renderer = PoseRenderer(module.gaussians)
        module.background = torch.zeros(3, device='cuda')
        module.motion = SpatialGatedMotion(motion_cfg)
        module.motion.setup(raw['xyz'].cuda(), raw['rotation'].cuda(), 2., frame_index=1)
        initial = copy.deepcopy(module.motion.frame_reference_state)
        with torch.no_grad():
            module.motion._hard_gate_frozen = True
            module.motion.region_translation[module.motion.owner[0], 0] = 2 * module.motion.steps[0]
            module.motion.gs_translation[3, 0] = 3 * module.motion.gs_steps[0]
            module.motion(motion_cfg.motion_steps - motion_cfg.amplitude_refit_steps + 1,
                          training=True)
            xyz, rotation = module.motion(motion_cfg.motion_steps)
            module.gaussians.apply_selected_motion(xyz, rotation)
        for parameter in module.motion.parameters():
            parameter.requires_grad_(False)
        module.opacity_update = QuantizedOpacityUpdate(config.opacity_update)
        module.opacity_update.setup(raw['opacity'].cuda())
        module.frame_prepared = module.motion_committed = True
        module.frame_finalized = False
        module.reference_count = 8
        module.initial_reference = dict(original_count=8, retained_count=8,
            removed_count=0, threshold=0.)
        module.birth_control = dict(pruned=0, split_added=0, previous_birth_count=0)
        module.retired_reference = None
        module.frame_report = None
        module.stage_image_metrics = []
        module._commit_opacity_and_retire()
        sources = torch.tensor([0, 5, 6])
        newborn = {name: value[sources].clone() for name, value in model_raw(module.gaussians).items()}
        newborn['xyz'][:, 0] += .003
        newborn['opacity'][:] = torch.logit(torch.tensor([[.95], [.98], [.02]], device='cuda'))
        module.births = self.gaussian(newborn, config.gaussian_stage2,
                                     births=True, sources=sources)
        module._freeze_base()
        Trainer._write_reference(module.trainer, 0, initial)
        return module, raw, initial

    @staticmethod
    def train_pose_step(module, step):
        module.trainer.global_step = step
        pose = module.forward(None, training=True)['render']
        old_count = len(module.gaussians._xyz)
        if module.birth_refit_active:
            expected, _ = module.motion(step)
            torch.testing.assert_close(pose[:old_count], expected, rtol=0, atol=0)
        (pose[:, 0] - .04).square().mean().backward()
        points = torch.zeros(len(pose), 3, device='cuda', requires_grad=True)
        points.grad = torch.zeros_like(points)
        package = dict(visibility_filter=torch.ones(len(pose), dtype=torch.bool, device='cuda'),
                       radii=torch.zeros(len(pose), device='cuda'), viewspace_points=points)
        module.post_training_step(package)

    def test_boundary_resume_fixed_support_final_pose_and_byte_exact_replay(self):
        with tempfile.TemporaryDirectory() as temporary:
            first_dir = Path(temporary) / 'original'
            module, raw, initial = self.active(first_dir)
            self.assertEqual(module.birth_selection_end_step, 260)
            self.assertEqual(len(module.gaussians._xyz), 8)  # No alpha deletion at motion end.
            self.train_pose_step(module, 260)
            self.assertTrue(module.birth_refit_active)
            self.assertEqual(len(module.births._xyz), 2)  # Weak .02 newborn lost competition.
            self.assertEqual(len(module.gaussians._xyz) + len(module.births._xyz), 8)
            source_ids = module.births.source_rows.clone()
            gates = module.motion.opened.clone()
            support = copy.deepcopy(module.motion._amplitude_refit_support)
            module.commit_birth_replacement()  # Idempotent.
            checkpoint = dict(global_step=260, current_frame=1)
            module.on_save_checkpoint(checkpoint)
            checkpoint = copy.deepcopy(checkpoint)
            resumed, _, _ = self.active(Path(temporary) / 'resumed')
            resumed.on_load_checkpoint(checkpoint)
            self.assertTrue(resumed.birth_refit_active)
            Trainer._write_reference(resumed.trainer, 0, initial)
            for step in range(261, 301):
                for worker in (module, resumed):
                    self.train_pose_step(worker, step)
                    self.assertEqual(len(worker.gaussians._xyz) + len(worker.births._xyz), 8)
                    self.assertTrue(torch.equal(worker.motion.opened, gates))
                    if step < 300:
                        self.assertTrue(torch.equal(worker.births.source_rows, source_ids))
                        for name, mask in support.items():
                            self.assertFalse(bool((worker.motion._amplitude_refit_support[name] & ~mask).any()))
                        self.assertFalse(bool(worker.motion.gs_translation[
                            ~support['gs_translation']].ne(0).any()))
            self.assertTrue(module.frame_finalized)
            self.assertTrue(resumed.frame_finalized)
            for worker in (module, resumed):
                stream = worker.output_dir / 'frame_storage' / 'rigid_tree'
                _, decoded = decode_frame(stream, 1, initial, raw)
                final = model_raw(worker.gaussians)
                for name in RAW_FIELDS:
                    torch.testing.assert_close(final[name].cpu(), decoded[name], rtol=0, atol=0)
                self.assertEqual(worker.frame_report['retired_count'], worker.frame_report['birth_count'])
            for suffix in ('motion_001.bin', 'birth_001.bin', 'frame_001.json'):
                self.assertEqual((first_dir / 'frame_storage/rigid_tree' / suffix).read_bytes(),
                    (resumed.output_dir / 'frame_storage/rigid_tree' / suffix).read_bytes())
            finalized = dict(global_step=300, current_frame=1)
            module.on_save_checkpoint(finalized)
            next_frame, _, _ = self.active(Path(temporary) / 'next_frame')
            next_frame.on_load_checkpoint(copy.deepcopy(finalized))
            self.assertTrue(next_frame.frame_finalized)
            self.assertFalse(next_frame.birth_refit_active)
            next_frame.datamodule.current_frame = 2
            next_frame.begin_next_frame()
            next_frame.prepare_frame()
            self.assertEqual(len(next_frame.gaussians._xyz), 8)
            self.assertEqual(next_frame.birth_control['previous_birth_count'], 2)


if __name__ == '__main__':
    unittest.main()
