"""Budget replacement must preserve appearance, causal motion, and replay."""

import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from pipeline.module import (BirthConfig, replace_low_opacity_gaussians,
                             restore_retired_reference)
from pipeline.model.birth_quantization import (RAW_FIELDS, RAW_ATTRIBUTES,
    QuantizedNewGaussianView, materialize_quantized_gaussians, model_raw)
from pipeline.model.rigid_residual_tree import RigidTreeConfig
from pipeline.model.rigid_tree_codec import DecodedState, decode_motion
from pipeline.model.rigid_tree_birth_codec import export_birth_attributes
from pipeline.model.spatial_gated_motion import SpatialGatedMotion
from pipeline.trainer import Trainer
from scripts.replay_rigid_tree import decode_frame, replay
from scripts.benchmark_rigid_residual_tree import _decode_frame


class Rows:
    def __init__(self, raw):
        self.prune_calls = 0
        for name, attr in zip(RAW_FIELDS, RAW_ATTRIBUTES):
            setattr(self, attr, raw[name].detach().clone())

    def prune_points(self, remove):
        self.prune_calls += 1
        for attr in RAW_ATTRIBUTES:
            setattr(self, attr, getattr(self, attr)[~remove].clone())


def reference_raw():
    xyz = torch.cartesian_prod(*[torch.tensor([-1., 1.])] * 3)
    ids = torch.arange(len(xyz), dtype=torch.float32)
    return dict(xyz=xyz, rotation=torch.tensor([[1., 0., 0., 0.]] * len(xyz)),
        logscale=ids[:, None].expand(-1, 3).clone() * .03,
        opacity=torch.logit(torch.tensor([[.005]] + [[.7]] * (len(xyz) - 1))),
        sh_dc=ids[:, None, None].expand(-1, 1, 3).clone() * .1,
        sh_rest=ids[:, None, None].expand(-1, 3, 3).clone() * .01)


def motion_model(raw):
    cfg = RigidTreeConfig()
    cfg.spatial_gate.max_depth = 2
    cfg.spatial_gate.min_region_size = 2
    cfg.spatial_gate.training_schedule = 'joint'
    cfg.spatial_gate.start_step = cfg.spatial_gate.gate_start_step = 0
    cfg.spatial_gate.initial_logit = -2.
    cfg.validate()
    motion = SpatialGatedMotion(cfg)
    motion.setup(raw['xyz'], raw['rotation'], 2., frame_index=1)
    return motion


def export_frame(directory, motion, raw, sources):
    xyz, rotation = motion.materialized_pose()
    base = {**raw, 'xyz': xyz, 'rotation': rotation}
    newborn = {name: value[sources].clone() for name, value in base.items()}
    if len(sources):
        newborn['xyz'][:, 0] += .003
        newborn['opacity'][:] = torch.logit(torch.tensor(.8))
        if len(sources) > 1:
            newborn['opacity'][-1] = torch.logit(torch.tensor(.001))
        newborn['sh_dc'] += .01
    view = (QuantizedNewGaussianView(Rows(newborn), BirthConfig(), 2., base, sources)
            if len(sources) else None)
    expected, _ = materialize_quantized_gaussians(base, view)
    motion.register_materialized_layout(expected['xyz'], expected['rotation'], sources)
    frame = motion.frame_index
    motion.export_motion(directory / f'motion_{frame:03d}.bin')
    expected['xyz'] = motion.codec_previous_state.xyz
    expected['rotation'] = motion.codec_previous_state.rotation
    if len(sources):
        export_birth_attributes(directory / f'birth_{frame:03d}.bin',
            {name: expected[name][-len(sources):] for name in RAW_FIELDS[2:]},
            len(raw['xyz']),
            symbols={name: view.symbols[name] for name in RAW_FIELDS[2:]},
            steps={name: view.steps[name] for name in RAW_FIELDS[2:]},
            predictor={name: view.predictor[name] for name in RAW_FIELDS[2:]},
            source_rows=sources)
    return expected


class GaussianReplacementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_replacement_retires_weak_old_and_new_rows_without_reordering(self):
        raw = reference_raw()
        model = Rows(raw)
        model._opacity[-1] = torch.logit(torch.tensor(.001))
        rows, retired, report = replace_low_opacity_gaussians(model, 2)
        torch.testing.assert_close(rows, torch.arange(1, 7))
        self.assertEqual(report['retired_older_count'], 1)
        self.assertEqual(report['retired_previous_birth_count'], 1)
        torch.testing.assert_close(retired['rows'], torch.tensor([0, 7]))
        torch.testing.assert_close(model._features_dc, raw['sh_dc'][rows])

    def test_empty_births_and_invalid_budgets_do_not_delete_points(self):
        model = Rows(reference_raw())
        for bad in (-1, 8, 9, True, 1.5):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                replace_low_opacity_gaussians(model, bad)
        rows, retired, report = replace_low_opacity_gaussians(model, 0)
        self.assertEqual(model.prune_calls, 0)
        self.assertIsNone(retired)
        self.assertEqual(report['retired_count'], 0)
        torch.testing.assert_close(rows, torch.arange(8))
        self.assertFalse(BirthConfig().replace_low_opacity)
        with self.assertRaisesRegex(ValueError, 'replace_low_opacity'):
            BirthConfig(replace_low_opacity=1).validate(100)

    def test_opacity_ties_have_deterministic_retirement_order(self):
        model = Rows(reference_raw())
        model._opacity.fill_(0.)
        rows, retired, _ = replace_low_opacity_gaussians(model, 2)
        torch.testing.assert_close(retired['rows'], torch.tensor([0, 1]))
        torch.testing.assert_close(rows, torch.arange(2, 8))

    def test_cross_frame_births_deletions_replay_and_midframe_bootstrap(self):
        # More births than deletions must still retire old appearance rows;
        # inferring birth count from the net change would silently be wrong.
        for new_count in (0, 1, 3):
            with self.subTest(new_count=new_count), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                raw = reference_raw()
                motion = motion_model(raw)
                initial = copy.deepcopy(motion.frame_reference_state)
                with torch.no_grad():
                    motion.gs_translation[:, 0] = torch.arange(8) * .002
                    motion.gs_rotation[:, 1] = torch.arange(8) * .001
                first = export_frame(directory, motion, raw, torch.tensor([0, 6]))
                prior, decoded_first = decode_frame(directory, 1, initial, raw)
                for name in RAW_FIELDS:
                    torch.testing.assert_close(decoded_first[name], first[name], rtol=0, atol=0)
                full = tuple(value.clone() for value in motion._full_motion_symbols())
                model = Rows(first)
                rows, retired, report = replace_low_opacity_gaussians(model, 2)
                self.assertEqual(report['retired_older_count'], 1)
                self.assertEqual(report['retired_previous_birth_count'], 1)
                kept = model_raw(model)
                motion.setup(kept['xyz'], kept['rotation'], 2., frame_index=2, previous_rows=rows)
                for actual, previous in zip(motion._full_motion_symbols(), full):
                    expected = torch.cat((previous, previous[torch.tensor([0, 6])]))[rows]
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                restored = SpatialGatedMotion(copy.deepcopy(motion.config))
                restored.restore(copy.deepcopy(motion.capture()))
                self.assertEqual(restored._survivor_mapping_bytes, motion._survivor_mapping_bytes)
                rebuilt = restore_retired_reference(kept, prior, rows, retired)
                for name in RAW_FIELDS:
                    torch.testing.assert_close(rebuilt[name], first[name], rtol=0, atol=0)
                bootstrap_module = SimpleNamespace(gaussians=model, motion=restored,
                    frame_finalized=False, retired_reference=retired,
                    initial_reference=dict(original_count=8, retained_count=8,
                                           removed_count=0, threshold=0.))
                worker = SimpleNamespace(output_dir=directory / 'resume', module=bootstrap_module,
                    datamodule=SimpleNamespace(cameras_extent=2.))
                Trainer._write_reference(worker, frame_index=1, state=prior)
                state_from_bootstrap, raw_from_bootstrap = replay(
                    worker.decoder_reference_path.parent, worker.decoder_reference_path, 1)
                self.assertEqual(state_from_bootstrap.digest(), prior.digest())
                for name in RAW_FIELDS:
                    torch.testing.assert_close(raw_from_bootstrap[name], first[name], rtol=0, atol=0)
                sources = torch.arange(new_count, dtype=torch.long)
                second = export_frame(directory, restored, kept, sources)
                for decode in (decode_frame, _decode_frame):
                    current, decoded_second = decode(directory, 2, prior, first)
                    torch.testing.assert_close(current.survivor_rows, rows)
                    self.assertEqual(len(current.xyz), 8 + new_count)
                    for name in RAW_FIELDS:
                        torch.testing.assert_close(decoded_second[name], second[name], rtol=0, atol=0)
                if new_count:
                    model = Rows(second)
                    rows, _, _ = replace_low_opacity_gaussians(model, new_count)
                    third_base = model_raw(model)
                    restored.setup(third_base['xyz'], third_base['rotation'], 2.,
                                   frame_index=3, previous_rows=rows)
                    third = export_frame(directory, restored, third_base,
                                         torch.empty(0, dtype=torch.long))
                    last, decoded_third = decode_frame(directory, 3, current, second)
                    self.assertEqual(len(last.xyz), 8)
                    for name in RAW_FIELDS:
                        torch.testing.assert_close(decoded_third[name], third[name], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
