"""Initial asset publishing and point-cloud overlay checks."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
import yaml

from scripts.derive_initial_cloud_overlay import derive_overlay
from scripts.publish_initial_asset import publish


class PublishInitialAssetTests(unittest.TestCase):
    @staticmethod
    def fixture(root: Path, sh_degree=1):
        root.mkdir(parents=True, exist_ok=True)
        count = 100_000
        coefficients = (sh_degree + 1) ** 2 - 1
        checkpoint = root / 'source-15000.pth'
        values = [sh_degree,
            torch.zeros(count, 3), torch.zeros(count, 1, 3),
            torch.zeros(count, coefficients, 3), torch.zeros(count, 3),
            torch.zeros(count, 4), torch.zeros(count, 1)]
        torch.save(dict(global_step=15_000, current_frame=0, gaussians=values),
                   checkpoint)
        run = root / 'layout-run'
        reference = run / 'frame_storage/rigid_tree/reference_frame0.npz'
        reference.parent.mkdir(parents=True)
        np.savez_compressed(reference,
            xyz=np.zeros((count, 3), np.float32),
            rotation=np.zeros((count, 4), np.float32),
            logscale=np.zeros((count, 3), np.float32),
            opacity=np.zeros((count, 1), np.float32),
            sh_dc=np.zeros((count, 1, 3), np.float32),
            sh_rest=np.zeros((count, coefficients, 3), np.float32),
            owner=np.zeros(count, np.int64), frame_index=np.int32(0),
            region_count=np.int32(8), initial_original_count=np.int64(count),
            initial_retained_count=np.int64(count), initial_removed_count=np.int64(0),
            initial_opacity_threshold=np.float64(.0075))
        config = dict(ckpt_path=str(checkpoint), trainer=dict(num_init_steps=15_000),
            data=dict(extra_dataset_kwargs=dict(num_frames=1)),
            module=dict(sh_degree=sh_degree,
                        rigid_tree=dict(spatial_gate=dict(reference_layout=None))))
        (run / 'config.yaml').write_text(yaml.safe_dump(config))
        return checkpoint, run

    def test_verified_pair_is_published_as_symlinks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint, run = self.fixture(root)
            output = root / 'asset'
            report = publish(checkpoint, run, output)
            self.assertEqual((report['global_step'], report['original_gaussians'],
                              report['retained_gaussians'], report['max_sh_degree']),
                             (15_000, 100_000, 100_000, 1))
            for name in ('ckpt-0-15000.pth', 'reference_frame0.npz',
                         'layout_config.yaml'):
                self.assertTrue((output / name).is_symlink())
            saved = json.loads((output / 'manifest.json').read_text())
            self.assertTrue(saved['link_only'])
            self.assertEqual(saved['checkpoint_sha256'], report['checkpoint_sha256'])
            with self.assertRaises(FileExistsError):
                publish(checkpoint, run, output)

    def test_sh3_pair_is_verified_without_weakening_layout_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint, run = self.fixture(root, sh_degree=3)
            report = publish(checkpoint, run, root / 'asset')
            self.assertEqual((report['active_sh_degree'],
                              report['max_sh_degree']), (3, 3))

            checkpoint, run = self.fixture(root / 'mismatch', sh_degree=3)
            config_path = run / 'config.yaml'
            config = yaml.safe_load(config_path.read_text())
            config['module']['sh_degree'] = 1
            config_path.write_text(yaml.safe_dump(config))
            with self.assertRaisesRegex(ValueError, 'SH degree'):
                publish(checkpoint, run, root / 'mismatched-asset')

    def test_layout_must_be_fresh_and_from_the_same_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint, run = self.fixture(root)
            config_path = run / 'config.yaml'
            config = yaml.safe_load(config_path.read_text())
            config['module']['rigid_tree']['spatial_gate']['reference_layout'] = 'old.npz'
            config_path.write_text(yaml.safe_dump(config))
            with self.assertRaisesRegex(ValueError, 'fresh spatial layout'):
                publish(checkpoint, run, root / 'asset')
            self.assertFalse((root / 'asset').exists())


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_cloud(path: Path, count: int) -> None:
    cloud = o3d.geometry.PointCloud()
    values = np.arange(count, dtype=np.float64)
    cloud.points = o3d.utility.Vector3dVector(
        np.stack([values, values % 3, values % 5], axis=1)
    )
    cloud.colors = o3d.utility.Vector3dVector(
        np.stack([values % 2, values % 4, values % 7], axis=1) / 7.0
    )
    normals = np.zeros((count, 3), dtype=np.float64)
    normals[:, 2] = 1.0
    cloud.normals = o3d.utility.Vector3dVector(normals)
    assert o3d.io.write_point_cloud(str(path), cloud)


class DeriveInitialCloudOverlayTest(unittest.TestCase):
    def test_links_source_and_replaces_only_point_cloud(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            source = tmp_path / "source"
            (source / "frames/0000").mkdir(parents=True)
            (source / "cam00/images").mkdir(parents=True)
            (source / "sparse/0").mkdir(parents=True)
            (source / "frames/0000/cam00.png").write_bytes(b"image")
            (source / "sparse/0/cameras.bin").write_bytes(b"cameras")
            (source / "sparse/0/images.bin").write_bytes(b"images")
            (source / "test_split.txt").write_text("cam00.png\n")
            (source / "adapter.json").write_text(json.dumps({"frames": 1}))
            source_cloud = source / "source_points.ply"
            _write_cloud(source_cloud, 8)
            (source / "points3D.ply").symlink_to(source_cloud)
            (source / "points3D_downsample2.ply").symlink_to("points3D.ply")
            (source / "sparse/0/points3D.ply").symlink_to("../../points3D.ply")

            fused = tmp_path / "fused.ply"
            _write_cloud(fused, 20)
            before = {
                path.relative_to(source): _sha(path)
                for path in source.rglob("*")
                if path.is_file() and not path.is_symlink()
            }
            output = tmp_path / "output/dense10"
            repo_link = tmp_path / "repo/data/dense10"
            manifest = derive_overlay(source, fused, output, repo_link, 10)

            self.assertTrue(repo_link.is_symlink())
            self.assertEqual(repo_link.resolve(), output.resolve())
            self.assertTrue((output / "frames").is_symlink())
            self.assertTrue((output / "cam00").is_symlink())
            self.assertEqual(
                (output / "sparse/0/cameras.bin").resolve(),
                (source / "sparse/0/cameras.bin").resolve(),
            )
            self.assertEqual(
                (output / "sparse/0/images.bin").resolve(),
                (source / "sparse/0/images.bin").resolve(),
            )
            self.assertEqual(
                (output / "sparse/0/points3D.ply").resolve(),
                (output / "points3D.ply").resolve(),
            )
            cloud = o3d.io.read_point_cloud(str(output / "sparse/0/points3D.ply"))
            self.assertEqual(len(cloud.points), 10)
            self.assertEqual(manifest["initial_points"], 10)
            self.assertFalse(manifest["source_images_copied"])
            self.assertEqual(json.loads((output / "adapter.json").read_text())["frames"], 1)
            after = {
                path.relative_to(source): _sha(path)
                for path in source.rglob("*")
                if path.is_file() and not path.is_symlink()
            }
            self.assertEqual(after, before)

    def test_refuses_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            source = tmp_path / "source"
            source.mkdir()
            fused = tmp_path / "fused.ply"
            fused.write_bytes(b"unused")
            output = tmp_path / "output"
            output.mkdir()
            with self.assertRaises(FileExistsError) as raised:
                derive_overlay(source, fused, output, tmp_path / "link", 10)
            self.assertEqual(raised.exception.args[0], output)


if __name__ == "__main__":
    unittest.main()
