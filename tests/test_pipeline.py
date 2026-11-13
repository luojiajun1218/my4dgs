"""检查训练管线的入口、数据和跨帧衔接。

覆盖管线加载、图像尺寸、首帧参考、数据路径迁移、检查点恢复和续训，
以及验证指标的状态管理；逐帧调度和属性更新由对应测试文件负责。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import copy
import importlib
import importlib.util
import json
import random
import shutil
import subprocess
import tempfile
import unittest
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image


# ----- test_pipeline_package -----
ROOT = Path(__file__).resolve().parents[1]


class ReferenceRelocationTests(unittest.TestCase):
    def test_only_verified_identical_layout_can_relocate_checkpoint_configuration(self):
        import hashlib
        from pipeline.continue_trainer import relocate_checkpoint_reference

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


    def test_libgs_loads_canonical_pipeline(self):
        from libgs.pipeline.loader import load_pipeline
        from pipeline import Config, Pipeline
        self.assertEqual((Config, Pipeline), load_pipeline('pipeline'))

    def test_cli_print_supports_default_and_canonical_pipeline_name(self):
        outputs = []
        for flags in ([], ['--pipeline=pipeline']):
            result = subprocess.run([sys.executable, 'main.py',
                '--config=config/n3dv_baseline_flame_steak_300frames_20261113.yaml',
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
        cls.extension = importlib.import_module('pipeline.continue_trainer')
        cls.rigid = importlib.import_module('pipeline.model.rigid_residual_tree')
        cls.spatial = importlib.import_module('pipeline.model.spatial_motion_training')
        cls.codec = importlib.import_module('pipeline.model.frame_stream_format')
        cls.birth = importlib.import_module('pipeline.model.frame_stream_format')
        cls.replay = importlib.import_module('pipeline.model.frame_stream_decode')
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
        config.proximal_gs_regularization = True
        gate = config.spatial_gate
        gate.training_schedule = 'joint'
        gate.learn_region_motion = True
        gate.start_step = gate.gate_start_step = 0
        gate.freeze_step = 180
        gate.region_motion = 'relative'
        gate.shared_image_recode = gate.joint_image_gradients = gate.shared_discovery = True
        gate.image_relaxation = 'stochastic_ste'
        gate.frame_training_plan = True
        config.validate()
        model = cls.spatial.SpatialMotionTraining(config)
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
        """构造已完成帧的码流和索引，不执行完整训练。"""
        raw = {k: v.clone() for k, v in raw.items()}
        stream = run / 'frame_storage' / 'rigid_tree'
        summary_dir = run / 'frame_summary'
        summary_dir.mkdir(exist_ok=True)
        for frame in range(start, end + 1):
            if model.frame_index != frame:
                model.setup(raw['xyz'], raw['rotation'], 1., frame_index=frame)
            # Construct a completed-frame fixture directly. This checks
            # continuation and replay, not the 200 optimizer steps themselves.
            model.prepare_plan_step(200)
            with torch.no_grad():
                model.gs_translation[:16, 0] = .012 + frame * .001
                model.gs_translation[16:24, 1] = .009 + frame * .001
            xyz, rotation = model.materialized_pose()
            model.register_materialized_layout(xyz, rotation)
            model.export_motion(stream / f'motion_{frame:03}.bin')
            decoded = model.codec_previous_state
            raw['xyz'], raw['rotation'] = decoded.xyz.clone(), decoded.rotation.clone()
            manifest = dict(version=1, frame=frame, optimization_steps=200,
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
            saving_ckpt_frames=[1, 2], saving_ckpt_steps=[200]))
        self.current_data = copy.deepcopy(self.old_data)
        self.current_data['extra_dataset_kwargs']['num_frames'] = 5
        self.current_trainer = copy.deepcopy(self.old_trainer)
        self.current_trainer['saving_ckpt_frames'] = [1, 2, 4]
        module_config = self.module.ModuleConfig(rigid_tree=self.template.config,
            sh_degree=1)
        module_config.birth.frame_training_plan = True
        module_config.full_update.enabled = True
        module_config.full_update.early_joint_repair = True
        module_config.full_update.dense_color_200 = True
        module_config.full_update.regional_full_200 = True
        gaussian_capture = (1, self.final_raw['xyz'].clone(), self.final_raw['sh_dc'].clone(),
            self.final_raw['sh_rest'].clone(), self.final_raw['logscale'].clone(),
            self.final_raw['rotation'].clone(), self.final_raw['opacity'].clone(),
            torch.zeros(len(self.final_raw['xyz'])), torch.zeros(len(self.final_raw['xyz']), 1),
            torch.zeros(len(self.final_raw['xyz']), 1), dict(state={}, param_groups=[]), 1.)
        report = json.loads((self.source / 'frame_summary/frame_002.json').read_text())
        self.checkpoint = dict(global_step=200, current_frame=2, rigid_tree_version=1,
            birth_allocation='frame_training_plan_v1',
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

    def test_recover_200_step_full_attribute_prefix_with_regional_packet(self):
        from pipeline.model.residual_gs_attributes_update import encode_full_attribute_delta
        from pipeline.model.regional_attributes_update import HEADER, MAGIC, decode_regional_full
        state = self.codec.DecodedState.restore(self.saved_motion['codec_previous_state'])
        nodes = len(state.parents)
        zeros = dict(logscale=torch.zeros(nodes, 3), sh_dc=torch.zeros(nodes, 1, 3),
                     sh_rest=torch.zeros(nodes, 3, 3))
        symbols = {key: torch.zeros_like(value) for key, value in zeros.items()}
        symbols['sh_dc'][int(state.owner[0]), 0, 0] = 3
        body = encode_full_attribute_delta(zeros, symbols, frame_index=2, steps=(.005, .001, .001))
        packet = HEADER.pack(MAGIC, 2, len(body), 0) + body
        stream = self.source / 'frame_storage/rigid_tree'
        (stream / 'regfull_002.bin').write_bytes(packet)
        raw = {key: value.clone() for key, value in self.final_raw.items()}
        raw.update(decode_regional_full(packet, raw, state.owner, nodes, 2))
        manifest_path = stream / 'frame_002.json'
        manifest = json.loads(manifest_path.read_text())
        manifest.update(optimization_steps=200, regional_full_attributes='regfull_002.bin',
                        regional_full_timing='after_point_attributes_before_births',
                        raw_state_sha256=self.birth.reference_digest(raw))
        manifest_path.write_text(json.dumps(manifest))
        self.checkpoint['birth_allocation'] = 'frame_training_plan_v1'
        self.checkpoint['global_step'] = 200
        self.checkpoint['rigid_tree_module_config']['full_update'].update(
            enabled=True, dense_color_200=True, regional_full_200=True)
        gaussian = list(self.checkpoint['gaussians'])
        for name, index in (('xyz', 1), ('sh_dc', 2), ('sh_rest', 3), ('logscale', 4),
                            ('rotation', 5), ('opacity', 6)):
            gaussian[index] = raw[name].clone()
        self.checkpoint['gaussians'] = tuple(gaussian)
        plan = self.plan()
        self.assertEqual(plan.resume_frame, 2)
        self.assertIn('frame_storage/rigid_tree/regfull_002.bin',
                      {self.entry_path(entry) for entry in plan.files})
        self.extension.copy_verified_prefix(plan)
        decoded, actual = self.replay.replay(self.destination/'frame_storage/rigid_tree',
            self.destination/'frame_storage/rigid_tree/reference_frame0.npz', through=2)
        self.assertEqual(decoded.frame_index, 2)
        self.assertEqual(self.birth.reference_digest(actual), self.birth.reference_digest(raw))

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


    def test_rejects_nonterminal_or_incomplete_checkpoints(self):
        for key, value in [('global_step', 199), ('current_frame', 1)]:
            bad = copy.deepcopy(self.checkpoint)
            bad[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.plan(checkpoint=bad)
        for flag in ('prepared', 'finalized', 'motion_committed'):
            bad = copy.deepcopy(self.checkpoint)
            bad['rigid_tree_frame'][flag] = False
            with self.subTest(flag=flag), self.assertRaises(ValueError):
                self.plan(checkpoint=bad)

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
        trainer['saving_ckpt_steps'] = [100, 200]
        with self.assertRaises(ValueError):
            self.plan(current_trainer_config=trainer)


    def test_real_trainer_loop_skips_completed_prefix_and_restores_rng_before_next_frame(self):
        calls = []

        class LoopHarness(self.trainer_module.Trainer):
            def frame_training_loop(trainer):
                self.assertEqual(trainer.global_step, 0)
                if trainer.current_frame == 3:
                    torch.testing.assert_close(torch.get_rng_state(), self.saved_rng['torch'], rtol=0, atol=0)
                calls.append(('train', trainer.current_frame))
                trainer.global_step = 200
                return dict(frame=trainer.current_frame)

        trainer = object.__new__(LoopHarness)
        trainer.config = self.trainer_module.TrainerConfig()
        trainer.module = SimpleNamespace(config=SimpleNamespace(rigid_tree=self.template.config),
            total_steps=200, datamodule=SimpleNamespace(current_frame=2, num_frames=5),
            begin_next_frame=lambda: calls.append(('begin', trainer.current_frame)))
        trainer.global_step = 200
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
                trainer.global_step = 200
                return dict(frame=trainer.current_frame)

        trainer = object.__new__(LoopHarness)
        trainer.config = self.trainer_module.TrainerConfig()
        trainer.module = SimpleNamespace(config=SimpleNamespace(rigid_tree=self.template.config),
            total_steps=200, datamodule=SimpleNamespace(current_frame=2, num_frames=5),
            begin_next_frame=lambda: calls.append(('begin', trainer.current_frame)))
        trainer.global_step = 200
        trainer._resume_rng = copy.deepcopy(self.saved_rng)
        trainer._end_frame_exclusive = limit(3, plan, 5)
        result = trainer.training_loop()
        self.assertEqual(calls, [('begin', 2), ('train', 3), ('begin', 3)])
        self.assertEqual([item['frame'] for item in result], [3])

    def test_rejects_missing_packet_index(self):
        for name in ('frame_storage/rigid_tree/motion_001.bin',
                     'frame_storage/rigid_tree/frame_001.json',
                     'frame_summary/frame_001.json'):
            path = self.source / name
            contents = path.read_bytes()
            path.unlink()
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.plan()
            path.write_bytes(contents)
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


    def test_disabled_lpips_does_not_construct_network(self):
        module = self.module(enabled=False)
        with patch.object(pipeline_module, 'LPIPS', side_effect=AssertionError('unexpected LPIPS')):
            metrics = module.validation_step(self.view, 0, 0)
        self.assertNotIn('lpips', metrics)
        self.assertFalse(hasattr(module, '_lpips_eval_cache'))


from pipeline.model import frame_stream_format as _stream_fmt
import pipeline.module as module
import pipeline.trainer as trainer
import pipeline.model.frame_stream_decode as replay__rigid_reference_pruning


class FakeNativeGaussians:
    def __init__(self):
        self.prune_calls = 0
        self.restore(self.raw(), None)

    @staticmethod
    def raw():
        centers = torch.arange(4, dtype=torch.float32).reshape(4, 1)
        return dict(_xyz=torch.cat((centers, torch.zeros(4, 2)), dim=1),
                    _rotation=torch.tensor([[1., 0., 0., 0.]] * 4),
                    _scaling=torch.zeros(4, 3),
                    _opacity=torch.logit(torch.tensor([[.01], [.2], [.005], [.5]])),
                    _features_dc=centers.reshape(4, 1, 1).expand(4, 1, 3).clone(),
                    _features_rest=torch.zeros(4, 0, 3))

    def capture(self):
        return {name: getattr(self, name).detach().clone()
                for name in module.RAW_ATTRIBUTES}

    def restore(self, packet, config):
        for name in module.RAW_ATTRIBUTES:
            setattr(self, name, torch.nn.Parameter(packet[name].detach().clone()))

    def prune_points(self, mask):
        self.prune_calls += 1
        for name in module.RAW_ATTRIBUTES:
            value = getattr(self, name)
            setattr(self, name, torch.nn.Parameter(value.detach()[~mask].clone()))


def fake_module(config, gaussians, frame, num_init_steps=15000):
    holder = SimpleNamespace(config=config, gaussians=gaussians,
        renderer=SimpleNamespace(gaussians=None), current_frame=frame,
        trainer=SimpleNamespace(config=SimpleNamespace(num_init_steps=num_init_steps)),
        _freeze_base=lambda: [getattr(gaussians, name).requires_grad_(False)
                              for name in module.RAW_ATTRIBUTES])
    return holder


class NativeReferencePruningTests(unittest.TestCase):
    def test_native_bootstrap_requires_exact_fixed_15000_step_protocol(self):
        config = module.ModuleConfig(initial_opacity_threshold=.015)
        native = FakeNativeGaussians()
        configured_11k = fake_module(
            config, native, frame=0, num_init_steps=11_000)
        with self.assertRaisesRegex(ValueError, 'exactly 15000 steps'):
            module.Module.on_load_checkpoint(configured_11k,
                dict(global_step=11_000, gaussians=native.capture()))

        configured_15k = fake_module(config, native, frame=0)
        with self.assertRaisesRegex(ValueError, '15000-step'):
            module.Module.on_load_checkpoint(configured_15k,
                dict(global_step=14_999, gaussians=native.capture()))

    def test_threshold_validation_and_full_delete_guard(self):
        for bad in (-.01, 1., float('nan'), float('inf'), True, '0.015'):
            with self.subTest(threshold=bad), self.assertRaisesRegex(
                    ValueError, 'initial_opacity_threshold'):
                module.validate_initial_opacity_threshold(bad)
        original = FakeNativeGaussians()
        with self.assertRaisesRegex(ValueError, 'every Gaussian'):
            module.prune_native_initial_gaussians(original, .9)
        self.assertEqual(original.prune_calls, 0)
        self.assertEqual(len(original._xyz), 4)
        unpruned = module.prune_native_initial_gaussians(original, 0.)
        self.assertEqual((unpruned['original_count'], unpruned['retained_count'],
                          unpruned['removed_count']), (4, 4, 0))
        self.assertEqual(original.prune_calls, 0)

    def test_native_prune_keeps_deterministic_rows_and_own_resume_does_not_prune(self):
        config = module.ModuleConfig(initial_opacity_threshold=.015)
        native = FakeNativeGaussians()
        active = fake_module(config, native, frame=0)
        module.Module.on_load_checkpoint(active,
            dict(global_step=15000, gaussians=native.capture()))
        torch.testing.assert_close(native._xyz[:, 0], torch.tensor([1., 3.]),
                                   rtol=0, atol=0)
        torch.testing.assert_close(native._features_dc[:, 0, 0],
                                   torch.tensor([1., 3.]), rtol=0, atol=0)
        self.assertEqual(native.prune_calls, 1)
        self.assertEqual(active.reference_count, 2)
        self.assertEqual(active.initial_reference, dict(
            source='native_frame0_checkpoint', threshold=.015,
            original_count=4, retained_count=2, removed_count=2))
        self.assertTrue(all(not getattr(native, name).requires_grad
                            for name in module.RAW_ATTRIBUTES))

        own = dict(gaussians=native.capture(), rigid_tree_version=1,
            rigid_tree_config=asdict(config.rigid_tree),
            rigid_tree_module_config=asdict(config),
            rigid_tree_initial_reference=active.initial_reference,
            rigid_tree_frame=dict(prepared=False, finalized=True,
                motion_committed=False, reference_count=2, report=None,
                birth_control={}), births=None)
        restored = FakeNativeGaussians()
        # The fixed bootstrap protocol must not make historical rigid-tree
        # checkpoints unreadable: their native first-frame step is no longer
        # consulted once the encoded dynamic state exists.
        resume = fake_module(config, restored, frame=1, num_init_steps=11_000)
        module.Module.on_load_checkpoint(resume, own)
        self.assertEqual(restored.prune_calls, 0)
        self.assertEqual(len(restored._xyz), 2)
        self.assertEqual(resume.initial_reference, active.initial_reference)
        self.assertEqual(resume.reference_count, 2)

    def test_pruned_reference_layout_and_frame_zero_replay_have_same_count(self):
        config = module.ModuleConfig(initial_opacity_threshold=.015)
        native = FakeNativeGaussians()
        active = fake_module(config, native, frame=0)
        module.Module.on_load_checkpoint(active,
            dict(global_step=15000, gaussians=native.capture()))
        with tempfile.TemporaryDirectory() as directory:
            worker = SimpleNamespace(output_dir=Path(directory), module=active,
                datamodule=SimpleNamespace(cameras_extent=1.))
            active.motion = importlib.import_module('pipeline.model.spatial_motion_training').SpatialMotionTraining(config.rigid_tree)
            trainer.Trainer._write_reference(worker, frame_index=0)
            with np.load(worker.decoder_reference_path, allow_pickle=False) as packet:
                self.assertEqual(int(packet['initial_original_count']), 4)
                self.assertEqual(int(packet['initial_retained_count']), 2)
                self.assertEqual(int(packet['initial_removed_count']), 2)
                self.assertEqual(float(packet['initial_opacity_threshold']), .015)
                self.assertEqual(len(_stream_fmt.DecodedState.from_reference(packet).owner), 2)
            state, raw = replay__rigid_reference_pruning.replay(worker.decoder_reference_path.parent,
                                       worker.decoder_reference_path, through=0)
            self.assertEqual(len(state.xyz), active.reference_count)
            torch.testing.assert_close(raw['xyz'][:, 0], torch.tensor([1., 3.]),
                                       rtol=0, atol=0)
            rebuilt = active.motion.codec_previous_state
            torch.testing.assert_close(state.owner, rebuilt.owner, rtol=0, atol=0)
            torch.testing.assert_close(state.parents, rebuilt.parents, rtol=0, atol=0)


def _state(frame):
    return _stream_fmt.DecodedState(
        xyz=torch.tensor([[0.25, -1., 2.]], dtype=torch.float32),
        rotation=torch.tensor([[1., 0., 0., 0.]], dtype=torch.float32),
        owner=torch.tensor([1], dtype=torch.int64),
        parents=torch.tensor([-1, 0], dtype=torch.int64),
        frame_index=frame,
        birth_sources=None,
        survivor_rows=torch.tensor([2], dtype=torch.int64) if frame else None)


class CheckpointStateRestoreTests(unittest.TestCase):
    def test_cpu_restore_preserves_values_and_dtypes(self):
        for frame in (0, 49):
            with self.subTest(frame=frame):
                original = _state(frame)
                restored = _stream_fmt.DecodedState.restore(original.capture())
                self.assertEqual(restored.frame_index, frame)
                for field in ('xyz', 'rotation', 'owner', 'parents', 'birth_sources', 'survivor_rows'):
                    before, after = getattr(original, field), getattr(restored, field)
                    if before is None:
                        self.assertIsNone(after)
                    else:
                        self.assertEqual(after.device.type, 'cpu')
                        self.assertEqual(after.dtype, before.dtype)
                        torch.testing.assert_close(after, before, rtol=0, atol=0)
                self.assertEqual(restored.digest(), original.digest())


from pipeline.continue_trainer import validate_missing_dataset_root


class DataResolutionTests(unittest.TestCase):
    def test_automatic_resolution_preserves_small_images_and_caps_large_images(self):
        from libgs.data.dataset import base

        with patch.object(base, 'warning'):
            for size, expected in (((1345, 1008), (1345, 1008)),
                                   ((2689, 2016), (1600, 1199))):
                with self.subTest(size=size):
                    dataset = SimpleNamespace(image_size=size, resolution=-1, resolution_scale=1.)
                    image = SimpleNamespace(width=size[0], height=size[1])
                    self.assertEqual(base.Dataset.resolve_size(dataset, image), expected)

    def test_explicit_resolution_uses_requested_scale(self):
        from libgs.data.dataset import base

        dataset = SimpleNamespace(image_size=(2689, 2016), resolution=2, resolution_scale=1.)
        image = SimpleNamespace(width=2689, height=2016)
        self.assertEqual(base.Dataset.resolve_size(dataset, image), (1344, 1008))


class DatasetRelocationTests(unittest.TestCase):
    def test_relocated_dataset_requires_matching_recorded_cameras_and_point_cloud(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, destination, data = root/'source', root/'destination', root/'data'
            source.mkdir(); destination.mkdir(); (data/'sparse/0').mkdir(parents=True)
            cameras = [dict(frame=frame, img_name='cam00', id=frame, fx=10.) for frame in range(2)]
            (source/'cameras.json').write_text(json.dumps(cameras))
            (destination/'cameras.json').write_text(json.dumps(cameras))
            (source/'input.ply').write_bytes(b'original point cloud')
            (data/'sparse/0/points3D.ply').write_bytes(b'original point cloud')
            validate_missing_dataset_root(source, destination, data, 2, 2)
            cameras[0]['fx'] = 11.
            (destination/'cameras.json').write_text(json.dumps(cameras))
            with self.assertRaisesRegex(ValueError, 'camera metadata'):
                validate_missing_dataset_root(source, destination, data, 2, 2)
            cameras[0]['fx'] = 10.
            (destination/'cameras.json').write_text(json.dumps(cameras))
            (data/'sparse/0/points3D.ply').write_bytes(b'different point cloud')
            with self.assertRaisesRegex(ValueError, 'point cloud'):
                validate_missing_dataset_root(source, destination, data, 2, 2)


if __name__ == "__main__":
    unittest.main()
