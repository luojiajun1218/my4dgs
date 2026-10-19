"""Protect temporal frame boundaries and existing data during preprocessing."""

import io
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

from PIL import Image

from scripts.prepare_n3dv_from_videos import prepare_scene, read_png


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


if __name__ == '__main__':
    unittest.main()
