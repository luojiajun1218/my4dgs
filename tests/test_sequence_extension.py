"""CPU-only prefix extension contracts and actual motion/codec continuation.

This file is independent of training imports. The production extension helper
is deliberately imported only when the suite is explicitly run.
"""

import copy
from dataclasses import asdict
import importlib
import json
from pathlib import Path
import random
import shutil
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
from PIL import Image
import torch



class SequenceExtensionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.extension = importlib.import_module('pipeline.ReCon-GS.sequence_extension')
        cls.rigid = importlib.import_module('pipeline.ReCon-GS.model.rigid_residual_tree')
        cls.spatial = importlib.import_module('pipeline.ReCon-GS.model.spatial_gated_motion')
        cls.codec = importlib.import_module('pipeline.ReCon-GS.model.rigid_tree_codec')
        cls.birth = importlib.import_module('pipeline.ReCon-GS.model.birth_quantization')
        cls.replay = importlib.import_module('scripts.replay_rigid_tree')
        cls.data_module = importlib.import_module('pipeline.ReCon-GS.data')
        cls.trainer_module = importlib.import_module('pipeline.ReCon-GS.trainer')
        cls.module = importlib.import_module('pipeline.ReCon-GS.module')
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


if __name__ == '__main__':
    unittest.main()
