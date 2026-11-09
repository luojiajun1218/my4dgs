"""Pipeline tests."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import copy
import importlib
import importlib.util
import json
import pickle
import random
import shutil
import subprocess
import tempfile
import time
import unittest
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from pipeline.sequence_extension import _validate_adaptive_birth_completion

# ----- test_pipeline_package -----
ROOT = Path(__file__).resolve().parents[1]


class ReferenceRelocationTests(unittest.TestCase):
    def test_only_verified_identical_layout_can_relocate_checkpoint_configuration(self):
        import hashlib
        from pipeline.sequence_extension import relocate_checkpoint_reference

        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'reference_frame0.npz'
            target.write_bytes(b'original reference layout')
            previous = '/old/location/reference_frame0.npz'
            info = {'file': target.name, 'original_path': previous,
                    'original_resolved_path': previous,
                    'sha256': hashlib.sha256(target.read_bytes()).hexdigest()}
            (target.parent / 'manifest.json').write_text(json.dumps({'files': {'reference_layout': info}}))
            saved = {'spatial_gate': {'reference_layout': previous, 'max_depth': 4}}
            checkpoint = {'rigid_tree_config': saved,
                          'rigid_tree_module_config': {'rigid_tree': saved},
                          'rigid_tree_motion': {'config': saved}}

            relocated = relocate_checkpoint_reference(checkpoint, str(target))
            for config in (relocated['rigid_tree_config'],
                           relocated['rigid_tree_module_config']['rigid_tree'],
                           relocated['rigid_tree_motion']['config']):
                self.assertEqual(config['spatial_gate']['reference_layout'], str(target))
                self.assertEqual(config['spatial_gate']['max_depth'], 4)
            self.assertEqual(checkpoint['rigid_tree_config']['spatial_gate']['reference_layout'], previous)

            unrelated = copy.deepcopy(checkpoint)
            unrelated['rigid_tree_config']['spatial_gate']['reference_layout'] = '/unrelated/reference.npz'
            self.assertIs(relocate_checkpoint_reference(unrelated, str(target)), unrelated)
            target.write_bytes(b'different layout')
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                relocate_checkpoint_reference(checkpoint, str(target))


class PipelinePackageTests(unittest.TestCase):
    def run_python(self, code):
        return subprocess.run([sys.executable, '-c', code], cwd=ROOT,
                              check=True, capture_output=True, text=True)

    def test_old_and_new_imports_share_classes_in_either_import_order(self):
        names = ['data', 'module', 'trainer', 'sequence_extension', 'model',
                 'model.gaussian', 'model.rigid_tree_codec',
                 'model.rigid_tree_birth_codec', 'model.legacy_opacity_codec',
                 'model.birth_quantization', 'model.rigid_residual_tree',
                 'model.spatial_gate', 'model.spatial_gated_motion',
                 'model.contribution_rasterizer', 'model.contribution_renderer']
        for legacy_first in (True, False):
            with self.subTest(legacy_first=legacy_first):
                self.run_python(f'''
import importlib
names = {names!r}
for name in names:
    canonical = 'pipeline.' + name
    for prefix in ('pipeline.ReCon-GS.', 'pipeline.my4dgs.'):
        first, second = (prefix + name, canonical) if {legacy_first!r} else (canonical, prefix + name)
        a = importlib.import_module(first)
        b = importlib.import_module(second)
        assert a is b, (first, second)
        assert a.__spec__.name == canonical, a.__spec__.name
''')

    def test_pure_model_import_does_not_load_the_training_entry(self):
        self.run_python('''
import importlib, sys
importlib.import_module('pipeline.model.rigid_tree_codec')
assert 'pipeline.entry' not in sys.modules
assert 'pipeline.module' not in sys.modules
''')

    def test_libgs_loads_both_names_and_historical_pickles(self):
        from libgs.pipeline.loader import load_pipeline
        config, pipeline = load_pipeline('my4dgs')
        self.assertEqual((config, pipeline), load_pipeline('ReCon-GS'))
        self.assertEqual((config, pipeline), load_pipeline('pipeline'))
        self.assertIs(pickle.loads(b'cpipeline.ReCon-GS\nConfig\n.'), config)
        codec = importlib.import_module('pipeline.model.rigid_tree_codec')
        self.assertIs(pickle.loads(
            b'cpipeline.ReCon-GS.model.rigid_tree_codec\nDecodedState\n.'), codec.DecodedState)

    def test_cli_print_supports_new_default_and_previous_pipeline_name(self):
        outputs = []
        for flags in ([], ['--pipeline=ReCon-GS']):
            result = subprocess.run([sys.executable, 'main.py',
                '--config=config/best_flame_steak.yaml',
                '--print', *flags], cwd=ROOT, check=True, capture_output=True, text=True)
            outputs.append(result.stdout)
        self.assertEqual(outputs[0], outputs[1])
        self.assertIn('initialization: robust_median', outputs[0])

    def test_cuda_extension_sources_resolve_after_the_directory_move(self):
        observed = importlib.import_module('pipeline.model.contribution_rasterizer')
        observed._backend.cache_clear()
        try:
            with patch('torch.utils.cpp_extension.load', return_value=object()) as load:
                observed._backend()
            kwargs = load.call_args.kwargs
            expected = (ROOT / 'extensions/observed_rasterizer').resolve()
            for source in kwargs['sources']:
                path = Path(source)
                self.assertTrue(path.is_relative_to(expected), source)
                self.assertTrue(path.is_file(), source)
            self.assertTrue(Path(kwargs['extra_include_paths'][0]).is_dir())
        finally:
            observed._backend.cache_clear()


# ----- test_sequence_extension -----
class SequenceExtensionTests(unittest.TestCase):
    def test_extension_rejects_truncated_training_frame_set(self):
        from pipeline.trainer import _validate_extension_frame_coverage

        items = [SimpleNamespace(frame=frame) for frame in (0, 0, 1, 1, 2, 2)]
        _validate_extension_frame_coverage(items, 3)
        with self.assertRaisesRegex(ValueError, 'missing=\\[2\\]'):
            _validate_extension_frame_coverage(items[:-2], 3)
        with self.assertRaisesRegex(ValueError, 'extra=\\[2\\]'):
            _validate_extension_frame_coverage(items, 2)
        with self.assertRaisesRegex(ValueError, 'invalid frame index'):
            _validate_extension_frame_coverage([SimpleNamespace(frame=None)], 3)

    @classmethod
    def setUpClass(cls):
        cls.extension = importlib.import_module('pipeline.sequence_extension')
        cls.rigid = importlib.import_module('pipeline.model.rigid_residual_tree')
        cls.spatial = importlib.import_module('pipeline.model.spatial_gated_motion')
        cls.codec = importlib.import_module('pipeline.model.rigid_tree_codec')
        cls.birth = importlib.import_module('pipeline.model.birth_quantization')
        cls.replay = importlib.import_module('scripts.replay_rigid_tree')
        cls.data_module = importlib.import_module('pipeline.data')
        cls.trainer_module = importlib.import_module('pipeline.trainer')
        cls.module = importlib.import_module('pipeline.module')
        cls.work = tempfile.TemporaryDirectory()
        cls.base = Path(cls.work.name)
        cls.dataset = cls.base / 'dataset'
        cls.images = []
        for frame in range(5):
            path = cls.dataset / 'frames' / f'{frame:04}' / 'cam00.png'
            path.parent.mkdir(parents=True)
            Image.new('RGB', (4, 4), (frame, 25, 50)).save(path)
            cls.images.append((frame, str(path)))
        torch.manual_seed(71)
        np.random.seed(72)
        random.seed(73)
        cls.initial_rng = cls.capture_rng()
        cls.template = cls.shared_fixture()
        cls.reference_raw = cls.initial_raw(cls.template)
        prefix = cls.base / 'fixture_prefix'
        prefix.mkdir()
        cls.write_reference(prefix, cls.template, cls.reference_raw)
        cls.final_raw = cls.run_frames(cls.template, 1, 2, prefix, cls.reference_raw)
        cls.saved_motion = copy.deepcopy(cls.template.capture())
        cls.saved_rng = cls.capture_rng()
        cls.prefix_bytes = {str(p.relative_to(prefix)): p.read_bytes()
                            for p in prefix.rglob('*') if p.is_file()}

    @classmethod
    def tearDownClass(cls):
        cls.work.cleanup()

    @staticmethod
    def capture_rng():
        return dict(torch=torch.get_rng_state().clone(), cuda=None,
                    numpy=np.random.get_state(), python=random.getstate())

    @staticmethod
    def restore_rng(state):
        torch.set_rng_state(state['torch'])
        np.random.set_state(state['numpy'])
        random.setstate(state['python'])

    @staticmethod
    def initial_raw(model):
        n = len(model.base_xyz)
        return dict(xyz=model.base_xyz.clone(), rotation=model.base_rotation.clone(),
            logscale=torch.full((n, 3), -3.), opacity=torch.zeros(n, 1),
            sh_dc=torch.zeros(n, 1, 3), sh_rest=torch.zeros(n, 3, 3))

    @classmethod
    def shared_fixture(cls):
        xyz = torch.cartesian_prod(torch.tensor([.1, .3, .6, .9]),
            torch.tensor([.1, .9]), torch.tensor([.1, .9])).repeat_interleave(2, 0)
        rotation = torch.zeros(len(xyz), 4); rotation[:, 0] = 1
        config = cls.rigid.RigidTreeConfig(motion_steps=200, birth_steps=100,
            motion_xyz_step_extent=.001, gs_xyz_step_extent=.001,
            motion_rotation_step_rad=.001, gs_rotation_step_rad=.001,
            rate_loss_weight=0., relative_motion_loss_weight=0.)
        model = cls.spatial.SpatialGatedMotion(config)
        model.setup(xyz, rotation, 1., frame_index=1)
        return model

    @classmethod
    def write_reference(cls, run, model, raw):
        stream = run / 'frame_storage' / 'rigid_tree'
        stream.mkdir(parents=True)
        state = model.frame_reference_state
        np.savez_compressed(stream / 'reference_frame0.npz',
            **{k: v.numpy() for k, v in raw.items()}, owner=state.owner.numpy(),
            parents=state.parents.numpy(), spatial_paths=state.spatial_paths.numpy(),
            spatial_centers=state.spatial_centers.numpy(),
            spatial_open=state.spatial_open.numpy(), frame_index=np.array(0),
            extent=np.array(model.extent), region_count=np.array(len(state.parents)))

    @classmethod
    def run_frames(cls, model, start, end, run, raw):
        """Actual 200-step motion updates; no renderer, births or opacity."""
        raw = {k: v.clone() for k, v in raw.items()}
        stream = run / 'frame_storage' / 'rigid_tree'
        summary_dir = run / 'frame_summary'
        summary_dir.mkdir(exist_ok=True)
        for frame in range(start, end + 1):
            if model.frame_index != frame:
                model.setup(raw['xyz'], raw['rotation'], 1., frame_index=frame)
            desired = model.base_xyz.clone()
            desired[:16, 0] += .012 + frame * .001
            desired[16:24, 1] += .009 + frame * .001
            for step in range(1, 201):
                # Exercise all three saved CPU RNGs, rather than comparing a
                # deterministic pose script which could ignore RNG restore.
                jitter = (float(torch.rand(())) + float(np.random.random()) + random.random()) * 1e-5
                model.observations.fill_(10)
                xyz, q = model.forward(step, training=True)
                loss = (xyz - desired - jitter).square().mean() + (q - model.base_rotation).square().mean()
                terms = model.loss_terms(step, False)
                loss = loss + terms['rate_loss'] + terms['relative_motion_loss']
                loss.backward()
                model.optimizer_step()
            model.record_final_diagnostics()
            xyz, rotation = model.materialized_pose()
            model.register_materialized_layout(xyz, rotation)
            model.export_motion(stream / f'motion_{frame:03}.bin')
            decoded = model.codec_previous_state
            raw['xyz'], raw['rotation'] = decoded.xyz.clone(), decoded.rotation.clone()
            manifest = dict(version=1, frame=frame,
                reference='frame_storage/rigid_tree/reference_frame0.npz',
                motion=f'motion_{frame:03}.bin', birth=None,
                old_count=len(raw['xyz']), birth_count=0, total_count=len(raw['xyz']),
                decoded_state_sha256=decoded.digest().hex(),
                raw_state_sha256=cls.birth.reference_digest(raw), no_old_deletions=True)
            (stream / f'frame_{frame:03}.json').write_text(json.dumps(manifest, indent=2))
            motion_bytes = (stream / f'motion_{frame:03}.bin').stat().st_size
            summary = dict(frame=frame, old_count=len(raw['xyz']), birth_count=0,
                total_count=len(raw['xyz']), motion=dict(bytes=motion_bytes), birth=None,
                opacity_update=None, encoded_bytes=motion_bytes,
                shared_reference_bytes=(stream / 'reference_frame0.npz').stat().st_size,
                decoder_reference='frame_storage/rigid_tree/reference_frame0.npz',
                frame_manifest=f'frame_storage/rigid_tree/frame_{frame:03}.json',
                materialized_sha256=cls.birth.reference_digest(raw))
            (summary_dir / f'frame_{frame:03}.json').write_text(json.dumps(summary, indent=2))
        return raw

    @staticmethod
    def camera_records(frames):
        return [dict(id=frame, frame=frame, img_name='cam00', width=4, height=4,
                     position=[0., 0., 0.], rotation=np.eye(3).tolist(), fx=3., fy=3.)
                for frame in range(frames)]

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source, self.destination = self.root / 'source', self.root / 'destination'
        self.source.mkdir()
        self.destination.mkdir()
        for name, value in self.prefix_bytes.items():
            path = self.source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(value)
        self.old_data = asdict(self.data_module.DataConfig(root=str(self.dataset),
            expected_image_size=[4, 4], extra_dataset_kwargs=dict(num_frames=3)))
        self.old_trainer = asdict(self.trainer_module.TrainerConfig(
            saving_ckpt_frames=[1, 2], saving_ckpt_steps=[300]))
        self.current_data = copy.deepcopy(self.old_data)
        self.current_data['extra_dataset_kwargs']['num_frames'] = 5
        self.current_trainer = copy.deepcopy(self.old_trainer)
        self.current_trainer['saving_ckpt_frames'] = [1, 2, 4]
        module_config = self.module.ModuleConfig(rigid_tree=self.template.config,
            sh_degree=1)
        gaussian_capture = (1, self.final_raw['xyz'].clone(), self.final_raw['sh_dc'].clone(),
            self.final_raw['sh_rest'].clone(), self.final_raw['logscale'].clone(),
            self.final_raw['rotation'].clone(), self.final_raw['opacity'].clone(),
            torch.zeros(len(self.final_raw['xyz'])), torch.zeros(len(self.final_raw['xyz']), 1),
            torch.zeros(len(self.final_raw['xyz']), 1), dict(state={}, param_groups=[]), 1.)
        report = json.loads((self.source / 'frame_summary/frame_002.json').read_text())
        self.checkpoint = dict(global_step=300, current_frame=2, rigid_tree_version=1,
            birth_allocation='clone_small_v1',
            rng_state=copy.deepcopy(self.saved_rng), rigid_tree_data_config=self.old_data,
            rigid_tree_trainer_config=self.old_trainer,
            rigid_tree_module_config=asdict(module_config), rigid_tree_config=asdict(self.template.config),
            gaussians=gaussian_capture, rigid_tree_motion=copy.deepcopy(self.saved_motion),
            rigid_tree_frame=dict(prepared=True, finalized=True, motion_committed=True,
                report=report, reference_count=32, birth_control={}, tree_statistics=[], stage_image_metrics=[]),
            births=None, old_opacity_delta=None, birth_source_rows=None,
            rigid_tree_initial_reference=dict(original_count=32, retained_count=32, removed_count=0, threshold=0.))
        (self.source / 'cameras.json').write_text(json.dumps(self.camera_records(3)))
        (self.destination / 'cameras.json').write_text(json.dumps(self.camera_records(5)))
        (self.source / 'config.yaml').write_text('old run config must stay intact\n')
        (self.destination / 'config.yaml').write_text('new run config must stay intact\n')

    def plan(self, **overrides):
        arguments = dict(checkpoint=self.checkpoint, current_data_config=self.current_data,
            current_trainer_config=self.current_trainer, source_run=self.source,
            destination_run=self.destination, current_extent=1., current_images=self.images)
        arguments.update(overrides)
        return self.extension.validate_sequence_extension(**arguments)

    @staticmethod
    def entry_path(entry):
        return str(entry['relative_path'] if isinstance(entry, dict) else entry.relative_path)

    def test_complete_prefix_copy_is_byte_identical_and_preserves_new_metadata(self):
        before = {str(p.relative_to(self.source)): p.read_bytes()
                  for p in self.source.rglob('*') if p.is_file()}
        plan = self.plan()
        self.assertEqual(plan.resume_frame, 2)
        paths = {self.entry_path(entry) for entry in plan.files}
        self.assertTrue(set(self.prefix_bytes).issubset(paths))
        self.assertNotIn('config.yaml', paths)
        self.assertNotIn('cameras.json', paths)
        provenance = self.extension.copy_verified_prefix(plan)
        self.assertEqual(provenance['identity_scope'], 'current_resolved_dataset_paths_no_historical_image_hash')
        for name, content in self.prefix_bytes.items():
            target = self.destination / name
            self.assertFalse(target.is_symlink())
            self.assertEqual(target.read_bytes(), content)
        self.assertEqual(before, {str(p.relative_to(self.source)): p.read_bytes()
                                 for p in self.source.rglob('*') if p.is_file()})
        self.assertEqual((self.destination / 'config.yaml').read_text(), 'new run config must stay intact\n')




    def test_extension_copies_full_attribute_packet_and_decodes_all_six_fields(self):
        from pipeline.model.full_attribute_update import FIELDS, encode_full_attribute_delta
        stream = self.source / 'frame_storage' / 'rigid_tree'
        reference = stream / 'reference_frame0.npz'
        previous, raw = self.replay.replay(stream, reference, through=1)
        attribute_reference = {name: raw[name] for name in FIELDS}
        symbols = {name: torch.zeros_like(value) for name, value in attribute_reference.items()}
        symbols['logscale'][0, 0] = 2
        symbols['sh_dc'][1, 0, 1] = -3
        symbols['sh_rest'][2, 1, 2] = 4
        packet = encode_full_attribute_delta(attribute_reference, symbols,
            frame_index=2, steps=(.005, .001, .001))
        (stream / 'fullattr_002.bin').write_bytes(packet)
        path = stream / 'frame_002.json'
        manifest = json.loads(path.read_text())
        diagnostic = 'frame_diagnostics/learnable_attribute_gate/frame_002.npz'
        diagnostic_path = self.source / diagnostic
        diagnostic_path.parent.mkdir(parents=True)
        np.savez_compressed(diagnostic_path, region_id=np.asarray([1]), selected=np.asarray([True]))
        manifest.update(full_update=True, full_attribute_update='fullattr_002.bin',
            full_attribute_update_timing='after_regional_dc_before_births',
            regional_attribute_repair=dict(diagnostic_sidecar=diagnostic))
        path.write_text(json.dumps(manifest, indent=2))
        decoded, expected = self.replay.decode_frame(stream, 2, previous, raw)
        manifest['raw_state_sha256'] = self.birth.reference_digest(expected)
        path.write_text(json.dumps(manifest, indent=2))
        capture = list(self.checkpoint['gaussians'])
        for name, index in dict(xyz=1, rotation=5, logscale=4, opacity=6, sh_dc=2, sh_rest=3).items():
            capture[index] = expected[name].clone()
        self.checkpoint['gaussians'] = tuple(capture)
        plan = self.plan()
        copied = {self.entry_path(entry) for entry in plan.files}
        self.assertIn('frame_storage/rigid_tree/fullattr_002.bin', copied)
        self.assertIn(diagnostic, copied)
        self.extension.copy_verified_prefix(plan)
        dest = self.destination / 'frame_storage' / 'rigid_tree'
        actual_state, actual = self.replay.replay(dest, dest / 'reference_frame0.npz', through=2)
        self.assertEqual(actual_state.digest(), decoded.digest())
        for name in ('xyz', 'rotation', 'logscale', 'opacity', 'sh_dc', 'sh_rest'):
            self.assertTrue(torch.equal(actual[name].view(torch.int32),
                                        expected[name].view(torch.int32)), name)
        self.assertEqual((self.destination / diagnostic).read_bytes(), diagnostic_path.read_bytes())

    def test_extension_accepts_only_an_equivalent_symlink_dataset_root(self):
        alias = self.root / 'dataset_alias'
        alias.mkdir()
        (alias / 'frames').symlink_to(self.dataset / 'frames', target_is_directory=True)
        config = copy.deepcopy(self.current_data)
        config['root'] = str(alias)
        images = [(frame, str(alias / 'frames' / f'{frame:04d}' / 'cam00.png'))
                  for frame in range(5)]
        plan = self.plan(current_data_config=config, current_images=images)
        self.assertEqual(plan.resume_frame, 2)
        self.assertTrue(all(Path(item['resolved_path']).is_relative_to(self.dataset)
                            for item in plan.images))
        changed = self.root / 'changed_dataset'
        for frame in range(5):
            target = changed / 'frames' / f'{frame:04d}' / 'cam00.png'
            target.parent.mkdir(parents=True)
            if frame == 1:
                Image.new('RGB', (4, 4), (100, 25, 50)).save(target)
            else:
                target.symlink_to(self.dataset / 'frames' / f'{frame:04d}' / 'cam00.png')
        config['root'] = str(changed)
        changed_images = [(frame, str(changed / 'frames' / f'{frame:04d}' / 'cam00.png'))
                          for frame in range(5)]
        with self.assertRaisesRegex(ValueError, 'changed a prefix image'):
            self.plan(current_data_config=config, current_images=changed_images)

    def test_explicit_completed_prefix_from_interrupted_longer_plan_is_verified(self):
        # Three completed frames, with an original four-frame plan. Never
        # rewrite the checkpoint to pretend the old run had another config.
        checkpoint = copy.deepcopy(self.checkpoint)
        checkpoint['rigid_tree_data_config']['extra_dataset_kwargs']['num_frames'] = 4
        (self.source / 'cameras.json').write_text(json.dumps(self.camera_records(4)))
        with self.assertRaises(ValueError):
            self.plan(checkpoint=checkpoint)
        before = copy.deepcopy(checkpoint)
        plan = self.plan(checkpoint=checkpoint, completed_prefix_frames=3)
        self.assertEqual((plan.resume_frame, plan.previous_frames, plan.source_planned_frames), (2, 3, 4))
        self.assertEqual({image['frame'] for image in plan.images}, {0, 1, 2})
        self.assertFalse(any('frame_003' in entry.relative_path for entry in plan.files))
        provenance = self.extension.copy_verified_prefix(plan)
        self.assertEqual(provenance['source_planned_frames'], 4)
        self.assertEqual(checkpoint['rigid_tree_data_config'], before['rigid_tree_data_config'])
        self.assertEqual((self.source / 'config.yaml').read_text(), 'old run config must stay intact\n')
        decoded, _ = self.replay.replay(self.destination / 'frame_storage/rigid_tree',
            self.destination / 'frame_storage/rigid_tree/reference_frame0.npz', through=2)
        self.assertEqual(decoded.digest().hex(), plan.decoded_sha256)

    def test_explicit_incomplete_prefix_can_recover_same_planned_length(self):
        checkpoint = copy.deepcopy(self.checkpoint)
        checkpoint['rigid_tree_data_config']['extra_dataset_kwargs']['num_frames'] = 4
        source_cameras = json.dumps(self.camera_records(4))
        (self.source / 'cameras.json').write_text(source_cameras)
        (self.destination / 'cameras.json').write_text(source_cameras)
        data = copy.deepcopy(self.current_data)
        data['extra_dataset_kwargs']['num_frames'] = 4
        trainer = copy.deepcopy(self.current_trainer)
        trainer['saving_ckpt_frames'] = [1, 2, 3]
        arguments = dict(checkpoint=checkpoint, current_data_config=data,
                         current_trainer_config=trainer)
        before = {str(p.relative_to(self.source)): p.read_bytes()
                  for p in self.source.rglob('*') if p.is_file()}
        plan = self.plan(**arguments, completed_prefix_frames=3)
        self.assertEqual((plan.previous_frames, plan.extended_frames,
                          plan.source_planned_frames, plan.resume_frame), (3, 4, 4, 2))
        with self.assertRaises(ValueError):
            self.plan(**arguments)
        with self.assertRaises(ValueError):
            self.plan(**arguments, completed_prefix_frames=4)
        bad = copy.deepcopy(checkpoint)
        bad['global_step'] -= 1
        with self.assertRaises(ValueError):
            self.plan(**{**arguments, 'checkpoint': bad}, completed_prefix_frames=3)
        self.extension.copy_verified_prefix(plan)
        stream = self.destination / 'frame_storage' / 'rigid_tree'
        decoded, raw = self.replay.replay(stream, stream / 'reference_frame0.npz', through=2)
        self.assertEqual(decoded.digest().hex(), plan.decoded_sha256)
        self.assertEqual(self.birth.reference_digest(raw), plan.raw_sha256)
        self.assertEqual(before, {str(p.relative_to(self.source)): p.read_bytes()
                                  for p in self.source.rglob('*') if p.is_file()})

    def test_explicit_prefix_length_cannot_bypass_completion_or_replay_checks(self):
        for length in (True, 0, 1, 4, 2.5):
            with self.subTest(length=length), self.assertRaises(ValueError):
                self.plan(completed_prefix_frames=length)
        with self.assertRaises(ValueError):
            self.plan(completed_prefix_frames=2)
        bad = copy.deepcopy(self.checkpoint)
        bad['global_step'] -= 1
        with self.assertRaises(ValueError):
            self.plan(checkpoint=bad, completed_prefix_frames=3)
        bad = copy.deepcopy(self.checkpoint)
        bad['gaussians'][1][0, 0] += 1
        with self.assertRaises(ValueError):
            self.plan(checkpoint=bad, completed_prefix_frames=3)

    def test_cpu_three_to_five_continuation_matches_parameters_rng_and_every_packet(self):
        self.extension.copy_verified_prefix(self.plan())
        continued = self.spatial.SpatialGatedMotion(self.template.config)
        continued.restore(copy.deepcopy(self.checkpoint['rigid_tree_motion']))
        self.restore_rng(copy.deepcopy(self.checkpoint['rng_state']))
        resumed_raw = self.run_frames(continued, 3, 4, self.destination, self.final_raw)
        resumed_rng = self.capture_rng()
        continuous_dir = self.root / 'continuous'
        continuous_dir.mkdir()
        self.restore_rng(copy.deepcopy(self.initial_rng))
        continuous = self.shared_fixture()
        self.write_reference(continuous_dir, continuous, self.reference_raw)
        continuous_raw = self.run_frames(continuous, 1, 4, continuous_dir, self.reference_raw)
        continuous_rng = self.capture_rng()
        for name in ('region_translation', 'region_rotation', 'gs_translation', 'gs_rotation'):
            torch.testing.assert_close(getattr(continued, name), getattr(continuous, name), rtol=0, atol=0)
        self.assertTrue(torch.equal(continued.owner, continuous.owner))
        torch.testing.assert_close(continued.opened, continuous.opened, rtol=0, atol=0)
        torch.testing.assert_close(resumed_rng['torch'], continuous_rng['torch'], rtol=0, atol=0)
        np.testing.assert_equal(resumed_rng['numpy'], continuous_rng['numpy'])
        self.assertEqual(resumed_rng['python'], continuous_rng['python'])
        for name in resumed_raw:
            torch.testing.assert_close(resumed_raw[name], continuous_raw[name], rtol=0, atol=0)
        for frame in range(1, 5):
            name = f'frame_storage/rigid_tree/motion_{frame:03}.bin'
            self.assertEqual((self.destination / name).read_bytes(), (continuous_dir / name).read_bytes())
        stream = self.destination / 'frame_storage/rigid_tree'
        decoded, raw = self.replay.replay(stream, stream / 'reference_frame0.npz', through=4)
        self.assertEqual(decoded.digest(), continued.codec_previous_state.digest())
        self.assertEqual(self.birth.reference_digest(raw), self.birth.reference_digest(resumed_raw))

    def test_rejects_nonterminal_or_incomplete_checkpoints(self):
        for key, value in [('global_step', 200), ('current_frame', 1)]:
            bad = copy.deepcopy(self.checkpoint)
            bad[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.plan(checkpoint=bad)
        for flag in ('prepared', 'finalized', 'motion_committed'):
            bad = copy.deepcopy(self.checkpoint)
            bad['rigid_tree_frame'][flag] = False
            with self.subTest(flag=flag), self.assertRaises(ValueError):
                self.plan(checkpoint=bad)

    def test_dynamic_birth_completed_prefix_cross_checks_actual_terminal_manifest(self):
        path = self.source/'frame_storage/rigid_tree/frame_002.json'
        original = json.loads(path.read_text())
        for allocation, index in (('residual_surface_cohorts_v4', 1), ('selected_population_refit_v5', 2),
                                  ('selected_population_refit_v5_1', 2), ('inherited_intrinsic_radiance_v6', 2),
                                  ('bounded_intrinsic_radiance_v6_1', 2), ('conservative_detail_family_v7', 2),
                                  ('surface_patch_takeover_v8',2), ('support_reincarnation_crossfit_v9',2),
                                  ('dual_predictive_support_v10',2),('source_residual_support_v11',2)):
            checkpoint = copy.deepcopy(self.checkpoint)
            checkpoint['birth_allocation'] = allocation
            checkpoint['rigid_tree_module_config']['birth']['mechanism'] = 'residual_transaction'
            checkpoint['rigid_tree_module_config']['birth']['residual']['max_cohorts'] = 2
            end = 200+100*(index+1)
            checkpoint['global_step'] = end
            checkpoint['birth_cohort_state'] = dict(index=index, start_step=end-100, end_step=end)
            checkpoint['birth_terminal_state'] = dict(active=False)
            path.write_text(json.dumps({**original, 'optimization_steps': end}))
            with self.subTest(allocation=allocation):
                self.assertEqual(self.plan(checkpoint=checkpoint).resume_frame, 2)
            for invalid in ('unfinished', 'wrong_step', 'wrong_manifest', 'active_terminal'):
                bad = copy.deepcopy(checkpoint)
                path.write_text(json.dumps({**original, 'optimization_steps': end}))
                if invalid == 'unfinished': bad['rigid_tree_frame']['finalized'] = False
                elif invalid == 'wrong_step': bad['global_step'] -= 100
                elif invalid == 'wrong_manifest': path.write_text(json.dumps({**original, 'optimization_steps': end-100}))
                elif allocation.startswith('selected_population_refit_v5') or allocation in (
                        'inherited_intrinsic_radiance_v6', 'bounded_intrinsic_radiance_v6_1',
                        'conservative_detail_family_v7','surface_patch_takeover_v8','support_reincarnation_crossfit_v9',
                        'dual_predictive_support_v10','source_residual_support_v11'):
                    bad['birth_terminal_state']['active'] = True
                else: continue
                with self.subTest(allocation=allocation, invalid=invalid), self.assertRaises(ValueError):
                    self.plan(checkpoint=bad)
        path.write_text(json.dumps(original))

    def test_single_quick300_completed_prefix_and_optimizer_ledger(self):
        checkpoint=copy.deepcopy(self.checkpoint)
        checkpoint['birth_allocation']='quick_source_residual_300_v12'
        checkpoint['rigid_tree_module_config']['birth'].update(mechanism='residual_transaction',radiance_mode='quick_source_residual')
        checkpoint['birth_cohort_state']=dict(index=0,start_step=200,end_step=300)
        checkpoint['quick_birth_state']=dict(phase=2,optimizer_steps=100,initialized=True)
        checkpoint['dual_birth_state']=None
        path=self.source/'frame_storage/rigid_tree/frame_002.json'
        manifest=json.loads(path.read_text())
        manifest['optimization_steps']=300
        manifest['training_step_accounting']=dict(motion=200,birth=100,total=300,birth_optimizer_updates=100,
            initializer_optimizer_updates=0,post_admission_optimizer_updates=0)
        path.write_text(json.dumps(manifest))
        self.assertEqual(self.plan(checkpoint=checkpoint).resume_frame,2)
        for failure in ('extra_cohort','unfinished250','wrong_update_count','hidden_refit','dual_state'):
            bad=copy.deepcopy(checkpoint);packet=copy.deepcopy(manifest)
            if failure=='extra_cohort':bad['birth_cohort_state'].update(index=1,start_step=300,end_step=400)
            elif failure=='unfinished250':bad['global_step']=250;bad['rigid_tree_frame']['finalized']=False
            elif failure=='wrong_update_count':bad['quick_birth_state']['optimizer_steps']=101
            elif failure=='hidden_refit':packet['training_step_accounting']['post_admission_optimizer_updates']=1
            else:bad['dual_birth_state']={'branches':{}}
            path.write_text(json.dumps(packet))
            with self.subTest(failure=failure),self.assertRaises(ValueError):self.plan(checkpoint=bad)

    def test_shadow_completed_prefix200_or300_requires_actual_population_ledger(self):
        path=self.source/'frame_storage/rigid_tree/frame_002.json'
        original=json.loads(path.read_text())
        for end,accepted in ((200,0),(300,6)):
            checkpoint=copy.deepcopy(self.checkpoint)
            checkpoint.update(birth_allocation='shadow_source_residual_300_v13',global_step=end,
                birth_cohort_state=dict(index=0,start_step=200,end_step=end),
                quick_birth_state=dict(phase=2 if accepted else 1,optimizer_steps=end-200,initialized=True))
            checkpoint['rigid_tree_module_config']['birth'].update(
                mechanism='residual_transaction',radiance_mode='shadow_source_residual')
            checkpoint['rigid_tree_frame']['birth_control']['shadow_initializer']=dict(accepted_points=accepted)
            manifest=copy.deepcopy(original)
            manifest.update(optimization_steps=end,training_step_accounting=dict(motion=200,birth=end-200,total=end,
                birth_optimizer_updates=end-200,initializer_optimizer_updates=0,post_admission_optimizer_updates=0))
            path.write_text(json.dumps(manifest))
            self.assertEqual(self.plan(checkpoint=checkpoint).resume_frame,2)
            for failure in ('wrong_population','wrong_updates','unfinished','extra_step','hidden_refit'):
                bad=copy.deepcopy(checkpoint);packet=copy.deepcopy(manifest)
                if failure=='wrong_population':bad['rigid_tree_frame']['birth_control']['shadow_initializer']['accepted_points']=0 if accepted else 3
                elif failure=='wrong_updates':bad['quick_birth_state']['optimizer_steps']+=1
                elif failure=='unfinished':bad['rigid_tree_frame']['finalized']=False
                elif failure=='extra_step':bad['global_step']=301
                else:packet['training_step_accounting']['post_admission_optimizer_updates']=1
                path.write_text(json.dumps(packet))
                with self.subTest(end=end,failure=failure),self.assertRaises(ValueError):self.plan(checkpoint=bad)

    def test_early70_shared_adam_prefix_participation_is_not_summed_as430(self):
        path=self.source/'frame_storage/rigid_tree/frame_002.json'
        base=json.loads(path.read_text())
        for end,selected in ((200,0),(300,1)):
            checkpoint=copy.deepcopy(self.checkpoint)
            checkpoint.update(birth_allocation='early_source_replacement_300_v15',global_step=end,
                birth_cohort_state=dict(index=0,start_step=70,end_step=end),
                early_replacement_state=dict(initialized=True,shared_steps=end,birth_updates=end-70))
            checkpoint['rigid_tree_module_config']['birth'].update(
                mechanism='residual_transaction',radiance_mode='early_source_replacement')
            checkpoint['rigid_tree_frame']['birth_control']['early_replacement']=dict(
                source_count=2,selected_population=selected,SGD_200_codec_replay_exact=True)
            manifest=copy.deepcopy(base)
            manifest.update(optimization_steps=end,training_step_accounting=dict(warm_gate=70,joint=130,
                final_population=end-200,global_iterations=end,shared_adam_steps=end,motion_participation=200,
                birth_participation=end-70,joint_overlap=130,extra_SGD_steps=0,
                topology_born_step=70,population_frozen_step=200))
            path.write_text(json.dumps(manifest));self.assertEqual(self.plan(checkpoint=checkpoint).resume_frame,2)
            for failure in ('newborn100','extra_shared_step','missing200_replay','extra_cohort','fake_warm'):
                bad=copy.deepcopy(checkpoint);packet=copy.deepcopy(manifest)
                if failure=='newborn100':bad['early_replacement_state']['birth_updates']=100
                elif failure=='extra_shared_step':bad['early_replacement_state']['shared_steps']=301
                elif failure=='missing200_replay':bad['rigid_tree_frame']['birth_control']['early_replacement']['SGD_200_codec_replay_exact']=False
                elif failure=='extra_cohort':bad['birth_cohort_state']['index']=1
                else:packet['training_step_accounting']['warm_gate']=69
                path.write_text(json.dumps(packet))
                with self.subTest(end=end,failure=failure),self.assertRaises(ValueError):self.plan(checkpoint=bad)

    def test_only_strict_frame_growth_and_future_checkpoint_appends_are_allowed(self):
        for frames in (2, 3):
            data = copy.deepcopy(self.current_data)
            data['extra_dataset_kwargs']['num_frames'] = frames
            with self.subTest(frames=frames), self.assertRaises(ValueError):
                self.plan(current_data_config=data)
        for changed in ('resolution', 'training_view_seed'):
            data = copy.deepcopy(self.current_data)
            data[changed] = 123
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                self.plan(current_data_config=data)
        for saves in ([2, 4], [1, 2, 0, 4]):
            trainer = copy.deepcopy(self.current_trainer)
            trainer['saving_ckpt_frames'] = saves
            with self.subTest(saves=saves), self.assertRaises(ValueError):
                self.plan(current_trainer_config=trainer)
        trainer = copy.deepcopy(self.current_trainer)
        trainer['saving_ckpt_steps'] = [200, 300]
        with self.assertRaises(ValueError):
            self.plan(current_trainer_config=trainer)

    def test_actual_cli_omegaconf_containers_obey_the_same_contract(self):
        from omegaconf import OmegaConf
        checkpoint = copy.deepcopy(self.checkpoint)
        checkpoint['rigid_tree_data_config'] = OmegaConf.create(self.old_data)
        checkpoint['rigid_tree_trainer_config'] = OmegaConf.create(self.old_trainer)
        plan = self.plan(checkpoint=checkpoint,
                         current_data_config=OmegaConf.create(self.current_data),
                         current_trainer_config=OmegaConf.create(self.current_trainer))
        self.assertEqual(plan.extended_frames, 5)

    def test_real_trainer_loop_skips_completed_prefix_and_restores_rng_before_next_frame(self):
        calls = []

        class LoopHarness(self.trainer_module.Trainer):
            def frame_training_loop(trainer):
                self.assertEqual(trainer.global_step, 0)
                if trainer.current_frame == 3:
                    torch.testing.assert_close(torch.get_rng_state(), self.saved_rng['torch'], rtol=0, atol=0)
                calls.append(('train', trainer.current_frame))
                trainer.global_step = 300
                return dict(frame=trainer.current_frame)

        trainer = object.__new__(LoopHarness)
        trainer.config = self.trainer_module.TrainerConfig()
        trainer.module = SimpleNamespace(config=SimpleNamespace(rigid_tree=self.template.config),
            total_steps=300, datamodule=SimpleNamespace(current_frame=2, num_frames=5),
            begin_next_frame=lambda: calls.append(('begin', trainer.current_frame)))
        trainer.global_step = 300
        trainer._resume_rng = copy.deepcopy(self.saved_rng)
        torch.manual_seed(999)
        result = trainer.training_loop()
        self.assertEqual(calls, [('begin', 2), ('train', 3), ('begin', 3), ('train', 4), ('begin', 4)])
        self.assertEqual([item['frame'] for item in result], [3, 4])
        self.assertIsNone(trainer._resume_rng)

    def test_prefix_frame_limit_trains_only_the_requested_future_frames(self):
        plan = self.plan()
        limit = self.trainer_module._extension_end_frame
        self.assertEqual(limit(None, None, 5), 5)
        self.assertEqual(limit(3, plan, 5), 4)
        for last, candidate in ((2, plan), (5, plan), (True, plan), (3, None)):
            with self.subTest(last=last), self.assertRaises(ValueError):
                limit(last, candidate, 5)
        calls = []

        class LoopHarness(self.trainer_module.Trainer):
            def frame_training_loop(trainer):
                calls.append(('train', trainer.current_frame))
                trainer.global_step = 300
                return dict(frame=trainer.current_frame)

        trainer = object.__new__(LoopHarness)
        trainer.config = self.trainer_module.TrainerConfig()
        trainer.module = SimpleNamespace(config=SimpleNamespace(rigid_tree=self.template.config),
            total_steps=300, datamodule=SimpleNamespace(current_frame=2, num_frames=5),
            begin_next_frame=lambda: calls.append(('begin', trainer.current_frame)))
        trainer.global_step = 300
        trainer._resume_rng = copy.deepcopy(self.saved_rng)
        trainer._end_frame_exclusive = limit(3, plan, 5)
        result = trainer.training_loop()
        self.assertEqual(calls, [('begin', 2), ('train', 3), ('begin', 3)])
        self.assertEqual([item['frame'] for item in result], [3])

    def test_rejects_missing_packet_index_or_required_birth_packet(self):
        for name in ('frame_storage/rigid_tree/motion_001.bin',
                     'frame_storage/rigid_tree/frame_001.json',
                     'frame_summary/frame_001.json'):
            path = self.source / name
            contents = path.read_bytes()
            path.unlink()
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.plan()
            path.write_bytes(contents)
        path = self.source / 'frame_storage/rigid_tree/frame_001.json'
        manifest = json.loads(path.read_text())
        manifest['birth'] = 'birth_001.bin'
        path.write_text(json.dumps(manifest))
        with self.assertRaises(ValueError):
            self.plan()

    def test_rejects_raw_hash_corruption_wrong_reference_and_checkpoint_pose(self):
        path = self.source / 'frame_storage/rigid_tree/frame_002.json'
        contents = path.read_bytes()
        for field, value in [('raw_state_sha256', '0' * 64),
                             ('decoded_state_sha256', '0' * 64),
                             ('reference', 'frame_storage/rigid_tree/reference_frame1.npz')]:
            manifest = json.loads(contents)
            manifest[field] = value
            path.write_text(json.dumps(manifest))
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.plan()
            path.write_bytes(contents)
        bad = copy.deepcopy(self.checkpoint)
        bad['gaussians'][1][0, 0] += .01
        with self.assertRaises(ValueError):
            self.plan(checkpoint=bad)

    def test_rejects_camera_frame_confusion_numeric_mismatch_missing_image_and_extent(self):
        path = self.destination / 'cameras.json'
        contents = path.read_bytes()
        records = json.loads(contents)
        records[1]['frame'] = 0  # Same cam00 name, distinct frame must not collapse.
        path.write_text(json.dumps(records))
        with self.assertRaises(ValueError):
            self.plan()
        records = json.loads(contents)
        records[1]['fx'] += .1
        path.write_text(json.dumps(records))
        with self.assertRaises(ValueError):
            self.plan()
        path.write_bytes(contents)
        images = list(self.images)
        images[1] = (0, images[1][1])
        with self.assertRaises(ValueError):
            self.plan(current_images=images)
        images = list(self.images)
        images[1] = (1, str(self.dataset / 'missing.png'))
        with self.assertRaises(ValueError):
            self.plan(current_images=images)
        with self.assertRaises(ValueError):
            self.plan(current_extent=1.01)

    def test_rejects_existing_destination_and_source_mutation_after_validation(self):
        plan = self.plan()
        target = self.destination / 'frame_storage/rigid_tree/motion_001.bin'
        target.parent.mkdir(parents=True)
        target.write_bytes(b'keep existing')
        with self.assertRaises((ValueError, FileExistsError)):
            self.extension.copy_verified_prefix(plan)
        self.assertEqual(target.read_bytes(), b'keep existing')
        shutil.rmtree(self.destination / 'frame_storage')
        plan = self.plan()
        source = self.source / 'frame_storage/rigid_tree/motion_001.bin'
        source.write_bytes(source.read_bytes() + b'tampered')
        with self.assertRaises(ValueError):
            self.extension.copy_verified_prefix(plan)
        self.assertTrue(source.read_bytes().endswith(b'tampered'), 'Copy refusal must not rewrite source')

    def test_source_and_destination_directory_links_and_nested_aliases_are_rejected(self):
        alias = self.root / 'source_alias'
        alias.symlink_to(self.source, target_is_directory=True)
        with self.assertRaises(ValueError):
            self.plan(source_run=alias)
        with self.assertRaises(ValueError):
            self.plan(destination_run=self.source)
        with self.assertRaises(ValueError):
            self.plan(destination_run=self.source / 'nested')
        plan = self.plan()
        outside = self.root / 'outside'
        outside.mkdir()
        (self.destination / 'frame_storage').symlink_to(outside, target_is_directory=True)
        with self.assertRaises((ValueError, FileExistsError)):
            self.extension.copy_verified_prefix(plan)
        self.assertEqual(list(outside.iterdir()), [])

    def test_leaf_symlink_is_read_only_dereferenced_into_regular_target(self):
        path = self.source / 'frame_storage/rigid_tree/motion_001.bin'
        contents = path.read_bytes()
        outside = self.root / 'external_motion.bin'
        outside.write_bytes(contents)
        path.unlink()
        path.symlink_to(outside)
        plan = self.plan()
        self.extension.copy_verified_prefix(plan)
        target = self.destination / 'frame_storage/rigid_tree/motion_001.bin'
        self.assertFalse(target.is_symlink())
        self.assertEqual(target.read_bytes(), contents)
        self.assertTrue(path.is_symlink())
        self.assertEqual(outside.read_bytes(), contents)


# ----- test_gpu_recovery -----
SPEC = importlib.util.spec_from_file_location(
    'gpu_recovery', Path(__file__).resolve().parents[1] / 'scripts/gpu_recover.py')
RECOVERY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RECOVERY)


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.state = self.root / 'state'
        self.state.mkdir()
        self.pending = self.state / 'pending.json'
        self.vendor = self.root / 'vendor'
        runtime = self.vendor / 'users/1000/runtime'
        runtime.mkdir(parents=True)
        (runtime / 'config.yaml').write_text('mixed-port: 7897\n')
        core = self.vendor / 'cores/verge-mihomo'
        core.parent.mkdir()
        core.write_text('test fixture, never executed')
        self.config = {'uid': 1000, 'user': 'gamer-02',
                       'proxy_state_dir': str(self.vendor),
                       'proxy_run_dir': str(self.root / 'run')}
        self.owner = {'owner_key': '1000', 'identity': {'Unix': {'uid': 1000, 'gid': 1000}}}
        self.desired = {'core_should_be_running': True, 'generation': 7,
                        'last_clash_config': {'core_config': {
                            'core_path': str(core), 'config_path': str(runtime / 'config.yaml'),
                            'config_dir': str(runtime),
                            'core_ipc_path': str(self.root / 'run/users/1000/verge-mihomo.sock')}}}
        self.pending_data = {'origin_boot_id': 'old-boot', 'proxy_snapshot': {
            'active_owner': self.owner, 'desired_state': self.desired}}
        self.patchers = [patch.object(RECOVERY, 'STATE', self.state),
                         patch.object(RECOVERY, 'PENDING', self.pending),
                         patch.object(RECOVERY, 'boot_id', return_value='new-boot')]
        for patcher in self.patchers:
            patcher.start()
            self.addCleanup(patcher.stop)

    def arm(self):
        RECOVERY.atomic_json(self.pending, self.pending_data)

    def test_current_boot_does_not_replay_pending_state(self):
        self.pending_data['origin_boot_id'] = 'new-boot'
        self.arm()
        self.assertFalse(RECOVERY.prepare_boot(self.config))
        self.assertFalse((self.vendor / 'active-owner.json').exists())

    def test_new_boot_recovers_owner_and_config_after_gui_cleanup(self):
        self.arm()
        self.assertTrue(RECOVERY.prepare_boot(self.config))
        self.assertEqual(RECOVERY.load_json(self.vendor / 'active-owner.json'), self.owner)
        desired = RECOVERY.load_json(self.vendor / 'users/1000/desired-state.json')
        self.assertTrue(desired['core_should_be_running'])
        self.assertEqual(desired['last_clash_config'], self.desired['last_clash_config'])
        self.assertEqual((self.vendor / 'active-owner.json').stat().st_mode & 0o777, 0o600)

    def test_valid_newer_config_is_preserved_and_replay_runs_once(self):
        self.arm()
        RECOVERY.atomic_json(self.vendor / 'active-owner.json', self.owner)
        current = {**self.desired, 'core_should_be_running': False, 'generation': 8}
        path = self.vendor / 'users/1000/desired-state.json'
        RECOVERY.atomic_json(path, current)
        self.assertTrue(RECOVERY.prepare_boot(self.config))
        self.assertEqual(RECOVERY.load_json(path)['generation'], 8)
        self.assertTrue(RECOVERY.load_json(path)['core_should_be_running'])
        RECOVERY.atomic_json(path, current)
        self.assertFalse(RECOVERY.prepare_boot(self.config))
        self.assertFalse(RECOVERY.load_json(path)['core_should_be_running'])

    def test_invalid_core_path_cannot_be_restored(self):
        self.pending_data['proxy_snapshot']['desired_state']['last_clash_config']['core_config']['core_path'] = '/tmp/unapproved-core'
        self.arm()
        with self.assertRaises(ValueError):
            RECOVERY.prepare_boot(self.config)
        self.assertFalse((self.vendor / 'active-owner.json').exists())

    def test_remote_ssh_readiness_does_not_require_a_working_gpu(self):
        status = dict(sshd_config_ok=True, ssh_banner_ok=True, default_route='wifi',
                      proxy_port_open=True, tailscale_running=True, tailscale_online=True,
                      gpu={'code': 1, 'output': 'GPU is lost'})
        self.assertTrue(RECOVERY.remote_ready(status))
        status['tailscale_online'] = False
        self.assertFalse(RECOVERY.remote_ready(status))

    def test_check_mode_never_arms_or_schedules_a_reboot(self):
        config_path = self.root / 'config.json'
        RECOVERY.atomic_json(config_path, self.config)
        with patch.object(RECOVERY, 'CONFIG', config_path), \
             patch.object(RECOVERY, 'ensure_root'), \
             patch.object(RECOVERY, 'status_now', return_value={'remote_checks_ready': True}), \
             patch.object(RECOVERY, 'print_status'), \
             patch.object(RECOVERY, 'start_recovery') as start:
            self.assertEqual(RECOVERY.main(['--check']), 0)
            start.assert_not_called()
        self.assertFalse(self.pending.exists())

    def test_collect_only_never_arms_or_schedules_a_reboot(self):
        config_path = self.root / 'config.json'
        RECOVERY.atomic_json(config_path, self.config)
        with patch.object(RECOVERY, 'CONFIG', config_path), \
             patch.object(RECOVERY, 'ensure_root'), \
             patch.object(RECOVERY, 'collect') as collect, \
             patch.object(RECOVERY, 'start_recovery') as start:
            self.assertEqual(RECOVERY.main(['--collect-only']), 0)
            collect.assert_called_once_with(self.config)
            start.assert_not_called()
        self.assertFalse(self.pending.exists())

    def test_hung_diagnostic_is_bounded(self):
        started = time.monotonic()
        result = RECOVERY.capture([sys.executable, '-c', 'import time; time.sleep(60)'], timeout=.1)
        self.assertTrue(result['timed_out'])
        self.assertEqual(result['code'], 124)
        self.assertLess(time.monotonic() - started, 4)

    def test_missing_diagnostic_tool_is_reported_without_stopping_recovery(self):
        result = RECOVERY.capture([str(self.root / 'nonexistent-tool')])
        self.assertEqual(result['code'], 127)
        self.assertFalse(result['timed_out'])

    def test_failed_shutdown_restores_the_previous_pending_record(self):
        self.arm()
        previous = RECOVERY.load_json(self.pending)

        def fake_capture(command, timeout=10):
            return {'code': 1 if command[0] == 'shutdown' else 0,
                    'output': 'simulated response', 'timed_out': False}

        with patch.object(RECOVERY, 'capture', side_effect=fake_capture), \
             patch.object(RECOVERY, 'boot_wifi_available', return_value=True), \
             patch.object(RECOVERY, 'snapshot_proxy', return_value=previous['proxy_snapshot']), \
             patch.object(RECOVERY, 'collect', return_value=self.root), \
             patch.object(RECOVERY.os, 'sync'):
            with self.assertRaisesRegex(RuntimeError, 'reboot request failed'):
                RECOVERY.start_recovery({**self.config, 'tailscale_ip': '100.0.0.1'})
        self.assertEqual(RECOVERY.load_json(self.pending), previous)

    def test_successful_recovery_saves_proxy_state_and_schedules_one_reboot(self):
        snapshot = self.pending_data['proxy_snapshot']
        with patch.object(RECOVERY, 'capture', return_value={'code': 0, 'output': ''}) as commands, \
             patch.object(RECOVERY, 'boot_wifi_available', return_value=True), \
             patch.object(RECOVERY, 'snapshot_proxy', return_value=snapshot), \
             patch.object(RECOVERY, 'collect', return_value=self.root), \
             patch.object(RECOVERY.os, 'sync'), patch('builtins.print'):
            RECOVERY.start_recovery({**self.config, 'tailscale_ip': '100.0.0.1'})
        pending = RECOVERY.load_json(self.pending)
        self.assertEqual(pending['proxy_snapshot'], snapshot)
        self.assertEqual(pending['origin_boot_id'], 'new-boot')
        shutdowns = [call.args[0] for call in commands.call_args_list if call.args[0][0] == 'shutdown']
        self.assertEqual(len(shutdowns), 1)
        self.assertEqual(shutdowns[0][:3], ['shutdown', '-r', '+1'])

    def test_reboot_is_refused_when_boot_wifi_is_not_configured(self):
        with patch.object(RECOVERY, 'capture', return_value={'code': 0}) as commands, \
             patch.object(RECOVERY, 'boot_wifi_available', return_value=False):
            with self.assertRaisesRegex(RuntimeError, 'Wi-Fi'):
                RECOVERY.start_recovery(self.config)
        self.assertEqual(commands.call_count, 1)
        self.assertEqual(commands.call_args.args[0], ['/usr/sbin/sshd', '-t'])
        self.assertFalse(self.pending.exists())

    def test_boot_report_marks_ssh_ready_even_if_gpu_is_still_lost(self):
        self.pending_data['diagnostics'] = str(self.root / 'diagnostics')
        self.arm()
        ready = {'remote_checks_ready': True, 'ssh_commands': ['ssh gamer-02@100.0.0.1']}
        with patch.object(RECOVERY, 'status_now', return_value=ready), \
             patch.object(RECOVERY, 'capture', return_value={'code': 1, 'output': 'GPU is lost'}), \
             patch.object(RECOVERY, 'print_status'):
            self.assertEqual(RECOVERY.boot_check(self.config), 0)
        report = RECOVERY.load_json(self.state / 'last-boot.json')
        self.assertTrue(report['remote_checks_ready'])
        self.assertEqual(report['gpu']['code'], 1)
        self.assertFalse(self.pending.exists())


def completed_adaptive_checkpoint(source=4, committed=2):
    updates = 230 if source else 0
    overlap = 130 if source else 0
    control = dict(source_count=source, training_population=source, population_cap=10,
                   low_alpha_newborns_removed=source-committed,
                   global_iterations=300, motion_iterations=200,
                   newborn_iterations=updates, joint_overlap_iterations=overlap,
                   final_population_iterations=100, new_alpha_iterations=updates,
                   population_screening=False, codec_trials200=0,
                   finish_render_calls=0, extra_SGD_steps=0)
    checkpoint = dict(rigid_tree_frame=dict(birth_control=dict(
        proposal_count=source, committed_population_count=committed,
        early_replacement=control)),
        early_replacement_state=dict(initialized=True, shared_steps=300,
                                     birth_updates=updates), dual_birth_state=None)
    manifest = dict(birth_count=committed, training_step_accounting=dict(
        warm_gate=70, joint=130, final_population=100, global_iterations=300,
        shared_adam_steps=300, motion_participation=200, birth_participation=updates,
        joint_overlap=overlap, extra_SGD_steps=0, topology_born_step=70,
        population_frozen_step=70, final_alpha_cleanup_step=300))
    return checkpoint, manifest


class AdaptiveCompletionTests(unittest.TestCase):
    def test_completed_no_screening_and_empty_population_are_accepted(self):
        for source, committed in ((4, 2), (4, 0), (0, 0)):
            with self.subTest(source=source, committed=committed):
                checkpoint, manifest = completed_adaptive_checkpoint(source, committed)
                _validate_adaptive_birth_completion(checkpoint, manifest, 300)

    def test_inconsistent_population_or_training_ledger_is_rejected(self):
        original, original_manifest = completed_adaptive_checkpoint()
        cases = (
            ('early_replacement_state', 'birth_updates', 229),
            ('early_replacement_state', 'shared_steps', 299),
            ('early_replacement', 'population_screening', True),
            ('early_replacement', 'training_population', 3),
            ('early_replacement', 'low_alpha_newborns_removed', 1),
            ('early_replacement', 'source_count', True),
            ('manifest', 'birth_count', 3),
            ('ledger', 'population_frozen_step', 200),
            ('ledger', 'extra_SGD_steps', 1),
        )
        for group, key, value in cases:
            with self.subTest(group=group, key=key):
                checkpoint, manifest = copy.deepcopy((original, original_manifest))
                if group == 'early_replacement_state':
                    checkpoint[group][key] = value
                elif group == 'early_replacement':
                    checkpoint['rigid_tree_frame']['birth_control'][group][key] = value
                elif group == 'manifest':
                    manifest[key] = value
                else:
                    manifest['training_step_accounting'][key] = value
                with self.assertRaises(ValueError):
                    _validate_adaptive_birth_completion(checkpoint, manifest, 300)

    def test_midframe_checkpoint_is_not_accepted(self):
        checkpoint, manifest = completed_adaptive_checkpoint()
        with self.assertRaises(ValueError):
            _validate_adaptive_birth_completion(checkpoint, manifest, 200)


# ----- LPIPS evaluation cache -----




import pipeline.module as pipeline_module


class _FakeLPIPS(torch.nn.Module):
    created = 0

    def __init__(self, network, version):
        super().__init__()
        if (network, version) != ('vgg', '0.1'):
            raise AssertionError('LPIPS network or weights changed')
        type(self).created += 1
        torch.rand(17)  # Real VGG construction also consumes the CPU RNG.
        self.weight = torch.nn.Parameter(torch.tensor(0.75), requires_grad=False)

    def forward(self, image, target):
        return (image - target).square().mean().reshape(1) * self.weight


class LPIPSEvaluationTests(unittest.TestCase):
    def setUp(self):
        _FakeLPIPS.created = 0
        self.image = torch.full((3, 16, 16), 0.25)
        self.view = SimpleNamespace(image=torch.full_like(self.image, 0.5))

    def module(self, enabled=True):
        config = pipeline_module.ModuleConfig(lpips_required=enabled, num_saving_images=0)
        module = pipeline_module.Module(config)
        module.forward = lambda viewpoint: {'render': self.image}
        return module

    def test_validation_reuses_metric_and_keeps_rng_and_state_dict(self):
        module = self.module()
        before_keys = set(module.state_dict())
        rng = torch.get_rng_state().clone()
        with patch.object(pipeline_module, 'LPIPS', _FakeLPIPS):
            first = module.validation_step(self.view, 0, 0)
            second = module.validation_step(self.view, 0, 0)
        self.assertEqual(_FakeLPIPS.created, 1)
        self.assertEqual(first, second)
        self.assertAlmostEqual(first['lpips'], 0.75 * 0.25**2)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(before_keys, set(module.state_dict()))

    def test_cold_and_warm_cache_leave_same_next_random_draw(self):
        with patch.object(pipeline_module, 'LPIPS', _FakeLPIPS):
            warm, cold = self.module(), self.module()
            warm.validation_step(self.view, 0, 0)
            rng = torch.get_rng_state().clone()
            warm.validation_step(self.view, 0, 0)
            after_warm = torch.rand(8)
            torch.set_rng_state(rng)
            cold.validation_step(self.view, 0, 0)
            after_cold = torch.rand(8)
        torch.testing.assert_close(after_warm, after_cold, rtol=0, atol=0)
        self.assertEqual(_FakeLPIPS.created, 2)

    def test_disabled_lpips_does_not_construct_network(self):
        module = self.module(enabled=False)
        with patch.object(pipeline_module, 'LPIPS', side_effect=AssertionError('unexpected LPIPS')):
            metrics = module.validation_step(self.view, 0, 0)
        self.assertNotIn('lpips', metrics)
        self.assertFalse(hasattr(module, '_lpips_eval_cache'))


if __name__ == "__main__":
    unittest.main()
