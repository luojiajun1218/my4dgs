"""Dataset preparation, CPU thermal supervision, package compatibility, and atomic output relocation."""
import io
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from PIL import Image
import os
import subprocess
import sys
import time
import importlib
import pickle
import threading
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.prepare_n3dv_from_videos import prepare_scene, read_png, reusable_images


class FragmentedPipe(io.BytesIO):
    def read(self, size=-1):
        return super().read(min(size, 3))


class N3DVPreparationTests(unittest.TestCase):
    def png(self, color):
        stream = io.BytesIO()
        Image.new('RGB', (5, 3), color).save(stream, format='PNG')
        return stream.getvalue()

    def test_fragmented_pipe_preserves_successive_camera_frames(self):
        first, second = self.png('red'), self.png('blue')
        pipe = FragmentedPipe(first + second)
        self.assertEqual(read_png(pipe), first)
        self.assertEqual(read_png(pipe), second)
        with self.assertRaises(EOFError):
            read_png(pipe)

    def test_truncated_or_invalid_frame_does_not_shift_the_sequence(self):
        with self.assertRaises(EOFError):
            read_png(io.BytesIO(self.png('red')[:-1]))
        with self.assertRaises(ValueError):
            read_png(io.BytesIO(b'notapng!'))

    def test_existing_directory_and_symlink_are_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            existing = root / 'existing'
            existing.mkdir()
            sentinel = existing / 'source_data'
            sentinel.write_text('original')
            link = root / 'link'
            link.symlink_to(existing, target_is_directory=True)
            for destination in (existing, link):
                with self.subTest(destination=destination), self.assertRaises(FileExistsError):
                    prepare_scene(root / 'missing_raw', destination, None)
            self.assertEqual(sentinel.read_text(), 'original')

    def test_resume_rejects_changed_inputs_before_modifying_the_scene(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw, destination = root / 'raw', root / 'prepared'
            raw.mkdir()
            destination.mkdir()
            for name in ('cam00', 'cam01', 'cam02'):
                (raw / (name + '.mp4')).touch()
            manifest = dict(raw=str(root / 'different_raw'), scene='raw', frames=300,
                            camera_names=['cam00.png', 'cam01.png', 'cam02.png'], status='failed')
            marker = destination / 'preprocessing_manifest.json'
            marker.write_text(json.dumps(manifest))
            before = marker.read_bytes()
            with self.assertRaisesRegex(ValueError, 'Resume inputs differ'):
                prepare_scene(raw, destination, SimpleNamespace(resume=True, frames=300))
            self.assertEqual(marker.read_bytes(), before)
            self.assertEqual([path.name for path in destination.iterdir()], [marker.name])

    def test_resume_never_follows_an_existing_scene_symlink(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / 'protected'
            target.mkdir()
            alias = root / 'alias'
            alias.symlink_to(target, target_is_directory=True)
            with self.assertRaises(FileExistsError):
                prepare_scene(root / 'raw', alias, SimpleNamespace(resume=True, frames=300))
            self.assertEqual(list(target.iterdir()), [])

    def test_resume_regenerates_a_frame_if_any_camera_png_is_invalid(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sizes = {'cam00.png': (5, 3), 'cam01.png': (5, 3)}
            paths = [root / Path(name).stem / 'images/0001.png' for name in sizes]
            for path in paths:
                path.parent.mkdir(parents=True)
                path.write_bytes(self.png('red'))
            self.assertTrue(reusable_images(root, 1, sizes))
            paths[1].write_bytes(self.png('blue')[:-1])
            self.assertFalse(reusable_images(root, 1, sizes))
            Image.new('RGB', (3, 5), 'blue').save(paths[1])
            self.assertFalse(reusable_images(root, 1, sizes))
            paths[1].unlink()
            paths[1].symlink_to(paths[0])
            self.assertFalse(reusable_images(root, 1, sizes))


class ThermalSupervisionTests(unittest.TestCase):
    def test_background_job_pauses_for_benchmark_and_resumes_after_flag_removed(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            supervision = directory / 'supervision'
            supervision.mkdir()
            sensor = directory / 'sensor'
            sensor.write_text('40000')
            status = supervision / 'preparation.json'
            flag = directory / 'pause_background_for_benchmark.json'
            child = subprocess.Popen([sys.executable, str(ROOT / 'scripts/run_cpu_limited.py'),
                '--cpus', str(min(os.sched_getaffinity(0))), '--sensor', str(sensor),
                '--status', str(status), '--interval=.02', '--', sys.executable,
                '-c', 'import time; time.sleep(60)'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            try:
                deadline = time.monotonic() + 5
                while read_status(status).get('status') != 'running' and time.monotonic() < deadline:
                    time.sleep(.02)
                self.assertEqual(read_status(status)['status'], 'running')
                flag.write_text(json.dumps(dict(owner_pid=os.getpid())))
                while read_status(status).get('status') != 'paused_for_benchmark' and time.monotonic() < deadline:
                    time.sleep(.02)
                self.assertEqual(read_status(status)['status'], 'paused_for_benchmark')
                flag.unlink()
                while read_status(status).get('status') != 'running' and time.monotonic() < deadline:
                    time.sleep(.02)
                self.assertEqual(read_status(status)['status'], 'running')
                self.assertGreater(read_status(status)['paused_seconds'], 0)
            finally:
                child.terminate()
                child.communicate(timeout=20)

    def test_next_job_reads_updated_core_and_library_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            sensor, status, policy, observed = (directory / name for name in
                ('sensor', 'status.json', 'policy.json', 'observed.json'))
            sensor.write_text('60000')
            cpus = sorted(os.sched_getaffinity(0))[:2]
            policy.write_text(json.dumps(dict(cpus=cpus, library_threads=2)))
            script = ('import json,os,pathlib; pathlib.Path(' + repr(str(observed)) +
                      ').write_text(json.dumps([sorted(os.sched_getaffinity(0)),os.environ["OMP_NUM_THREADS"]]))')
            result = subprocess.run([sys.executable, str(ROOT / 'scripts/run_cpu_limited.py'),
                '--cpus', str(cpus[0]), '--sensor', str(sensor), '--status', str(status),
                '--policy', str(policy), '--interval=.02', '--', sys.executable, '-c', script],
                capture_output=True, text=True, timeout=5)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(observed.read_text()), [cpus, '2'])

    def test_child_is_cpu_limited_and_pauses_until_temperature_recovers(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            sensor, status, observed = (directory / name for name in ('sensor', 'status.json', 'affinity.json'))
            sensor.write_text('40000')
            cpu = min(os.sched_getaffinity(0))
            script = 'import json,os,pathlib,time; pathlib.Path(' + repr(str(observed)) + ').write_text(json.dumps(sorted(os.sched_getaffinity(0)))); time.sleep(1)'
            child = subprocess.Popen([sys.executable, str(ROOT / 'scripts/run_cpu_limited.py'),
                '--cpus', str(cpu), '--sensor', str(sensor), '--status', str(status), '--interval=.02',
                '--', sys.executable, '-c', script], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            try:
                deadline = time.monotonic() + 5
                while not observed.exists() and time.monotonic() < deadline:
                    time.sleep(.02)
                self.assertEqual(json.loads(observed.read_text()), [cpu])
                sensor.write_text('75000')
                while read_status(status).get('status') != 'paused_for_cooling' and time.monotonic() < deadline:
                    time.sleep(.02)
                self.assertEqual(read_status(status)['status'], 'paused_for_cooling')
                sensor.write_text('65000')
                time.sleep(.06)
                self.assertEqual(read_status(status)['status'], 'paused_for_cooling')
                sensor.write_text('50000')
                stdout, stderr = child.communicate(timeout=5)
                self.assertEqual(child.returncode, 0, stderr.decode())
                final = read_status(status)
                self.assertEqual(final['status'], 'complete')
                self.assertEqual(final['pause_count'], 1)
                self.assertGreater(final['paused_seconds'], 0)
            finally:
                if child.poll() is None:
                    child.terminate()
                    child.communicate(timeout=20)


def read_status(path):
    return json.loads(path.read_text()) if path.exists() else {}


class PipelinePackageTests(unittest.TestCase):
    def test_output_relocation_keeps_concurrent_writers_and_open_files_valid(self):
        from scripts.n3dv_output_layout import scene_output
        with tempfile.TemporaryDirectory() as temporary:
            batch = Path(temporary) / 'output/n3dv/_batch'
            alias = batch / 'cook_spinach'
            directory = alias / 'frames'
            directory.mkdir(parents=True)
            anchor = alias / 'anchor'
            anchor.write_text('unchanged')
            ready = threading.Event()
            errors = []

            def writer():
                try:
                    for index in range(1000):
                        item = directory / 'pending'
                        item.write_text(str(index))
                        item.replace(directory / f'{index}.txt')
                        if index == 100:
                            ready.set()
                except BaseException as error:
                    errors.append(error)

            with anchor.open() as opened:
                identity = anchor.stat().st_ino
                thread = threading.Thread(target=writer)
                thread.start()
                self.assertTrue(ready.wait(5))
                target = scene_output(batch, 'cook_spinach', '20261020T000000')
                thread.join(5)
                self.assertFalse(thread.is_alive())
                self.assertEqual(errors, [])
                self.assertEqual(opened.read(), 'unchanged')
                self.assertEqual((target / 'anchor').stat().st_ino, identity)
                self.assertTrue(anchor.samefile(target / 'anchor'))
                self.assertEqual(len(list((target / 'frames').glob('*.txt'))), 1000)
                self.assertEqual(scene_output(batch, 'cook_spinach', 'unused'), target)

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
                '--config=config/n3dv/flame_steak_1020_full300.yaml',
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
            expected = ROOT / 'extensions/observed_rasterizer'
            for source in kwargs['sources']:
                path = Path(source)
                self.assertTrue(path.is_relative_to(expected), source)
                self.assertTrue(path.is_file(), source)
            self.assertTrue(Path(kwargs['extra_include_paths'][0]).is_dir())
        finally:
            observed._backend.cache_clear()


if __name__ == '__main__':
    unittest.main()
