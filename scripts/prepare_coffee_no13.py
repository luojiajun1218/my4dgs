"""Exclude the known asynchronous coffee camera; rebuild dense frame-zero seeds.

Keep the existing undistorted images and fixed camera calibration. Images are
symlinked, never copied or shifted. COLMAP removes cam13 and its sparse tracks;
PatchMatch and fusion start fresh, without old depth maps or Gaussian weights.
"""

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image

from libgs.data.utils.colmap import read_extrinsics_binary, read_intrinsics_binary


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--initial-points', type=int, default=40000)
    args = parser.parse_args()
    source = args.source.resolve()
    root = args.output.absolute()
    old_dense = source / 'reconstruction/dense'
    old_dataset = source / 'dataset'
    original = read_extrinsics_binary(old_dense / 'sparse/images.bin')
    original_names = sorted(im.name for im in original.values())
    if len(original_names) != 18 or original_names[12] != 'cam13.png':
        raise ValueError('Unexpected original camera order')
    names = [name for name in original_names if name != 'cam13.png']
    poses_bounds = np.load(old_dataset / 'poses_bounds.npy')
    if poses_bounds.shape != (18, 17):
        raise ValueError(f'Unexpected pose array: {poses_bounds.shape}')
    root.mkdir(parents=True, exist_ok=False)
    dense = root / 'reconstruction/dense'
    (dense / 'sparse').mkdir(parents=True)
    (dense / 'images').mkdir()
    (dense / 'stereo').mkdir()
    for name in ('depth_maps', 'normal_maps', 'consistency_graphs'):
        (dense / 'stereo' / name).mkdir()
    excluded = root / 'excluded_images.txt'
    excluded.write_text('cam13.png\n')
    manifest = dict(
        created_local_time=datetime.now().astimezone().isoformat(),
        source=str(source), excluded_cameras=['cam13.png'], cameras=names,
        frames=300, training_cameras=16, test_camera='cam00.png',
        temporal_shift_applied=False, source_images_copied=False,
        calibration='reuse fixed undistorted calibration; remove cam13 and sparse observations',
        initialization='fresh 17-view geometric PatchMatch, fusion, farthest-point sampling',
        official_issue='https://github.com/facebookresearch/Neural_3D_Video/issues/5',
        old_gaussian_checkpoint_loaded=False, commands=[], status='prepare',
    )

    def save():
        (root / 'preparation_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')

    def run(stage, command):
        manifest['status'] = stage
        manifest['commands'].append(command)
        save()
        print('STAGE', stage, flush=True)
        env = dict(os.environ, QT_QPA_PLATFORM='offscreen', OMP_NUM_THREADS='8')
        with (root / 'reconstruction' / f'{stage}.log').open('wb') as log:
            subprocess.run(command, check=True, env=env, stdout=log, stderr=subprocess.STDOUT)
        # COLMAP's worker errors can be logged while its process returns zero.
        lines = (root / 'reconstruction' / f'{stage}.log').read_text(errors='replace').splitlines()
        if any(line.startswith(('E20', 'F20')) for line in lines):
            raise RuntimeError(f'COLMAP reported errors in {stage}.log')

    save()
    try:
        run('remove_cam13', ['colmap', 'image_deleter', '--input_path', str(old_dense / 'sparse'),
            '--output_path', str(dense / 'sparse'), '--image_names_path', str(excluded)])
        filtered = read_extrinsics_binary(dense / 'sparse/images.bin')
        if sorted(im.name for im in filtered.values()) != names:
            raise ValueError('COLMAP camera filtering failed')
        for image_id, image in filtered.items():
            old = original[image_id]
            if (image.name != old.name or image.camera_id != old.camera_id or
                    not np.array_equal(image.qvec, old.qvec) or
                    not np.array_equal(image.tvec, old.tvec)):
                raise ValueError('Retained camera calibration changed')
        run('analyze_filtered_sparse', ['colmap', 'model_analyzer', '--path', str(dense / 'sparse')])
        for name in names:
            (dense / 'images' / name).symlink_to(old_dense / 'images' / name)
        (dense / 'stereo/patch-match.cfg').write_text(''.join(f'{name}\n__auto__, 20\n' for name in names))
        (dense / 'stereo/fusion.cfg').write_text('\n'.join(names) + '\n')
        points = root / 'reconstruction/points3D_downsample2.ply'
        half = root / 'dataset'
        full = root / 'dataset_full_first'
        for dataset in (half, full):
            (dataset / 'frames').mkdir(parents=True)
            adapter = dataset / 'sparse/0'
            adapter.mkdir(parents=True)
            (adapter / 'images.bin').symlink_to(dense / 'sparse/images.bin')
            (dataset / 'points3D.ply').symlink_to(points)
            (adapter / 'points3D.ply').symlink_to('../../points3D.ply')
            (dataset / 'points3D_downsample2.ply').symlink_to('points3D.ply')
            (dataset / 'test_split.txt').write_text('cam00.png\n')
        (half / 'sparse/0/cameras.bin').symlink_to(old_dataset / 'sparse/0/cameras.bin')
        (full / 'sparse/0/cameras.bin').symlink_to(dense / 'sparse/cameras.bin')
        (full / 'frames/0000').symlink_to(dense / 'images', target_is_directory=True)
        keep = [original_names.index(name) for name in names]
        np.save(half / 'poses_bounds.npy', poses_bounds[keep])
        half_camera = next(iter(read_intrinsics_binary(half / 'sparse/0/cameras.bin').values()))
        full_camera = next(iter(read_intrinsics_binary(full / 'sparse/0/cameras.bin').values()))
        for frame_index in range(300):
            frame = half / 'frames' / f'{frame_index:04d}'
            frame.mkdir()
            for name in names:
                src = old_dataset / 'frames' / frame.name / name
                if not src.is_file():
                    raise FileNotFoundError(src)
                with Image.open(src) as image:
                    if image.size != (half_camera.width, half_camera.height) or image.mode != 'RGB':
                        raise ValueError(f'Invalid half-size frame: {src}')
                (frame / name).symlink_to(src)
        for name in names:
            camera_frames = half / Path(name).stem / 'images'
            camera_frames.mkdir(parents=True)
            for frame_index in range(300):
                (camera_frames / f'{frame_index:04d}.png').symlink_to(
                    half / 'frames' / f'{frame_index:04d}' / name)
        adapter_note = 'images.bin retains full-resolution 2D tracks; half-size adapter is for loading, not SfM'
        manifest.update(
            filtered_extrinsics_sha256=sha256(dense / 'sparse/images.bin'),
            original_extrinsics_sha256=sha256(old_dense / 'sparse/images.bin'),
            retained_extrinsics_bit_exact=True, half_image_size=[half_camera.width, half_camera.height],
            full_image_size=[full_camera.width, full_camera.height],
            half_camera_sha256=sha256(half / 'sparse/0/cameras.bin'),
            pose_rows=keep, linked_frame_images=300 * len(names), adapter_note=adapter_note)
        for dataset in (half, full):
            (dataset / 'adapter.json').write_text(json.dumps(manifest, indent=2) + '\n')
        link_root = Path(__file__).resolve().parents[1] / 'data/colmap4d'
        for name, target in [('coffee_martini_half_no13', half), ('coffee_martini_full_first_no13', full)]:
            link = link_root / name
            if link.exists() or link.is_symlink():
                raise FileExistsError(link)
            link.symlink_to(target, target_is_directory=True)
        print('INPUT_VERIFIED 17 cameras, 300 frames, 5100 original-image links', flush=True)
        run('patch_match', ['colmap', 'patch_match_stereo', '--workspace_path', str(dense),
            '--workspace_format', 'COLMAP', '--PatchMatchStereo.geom_consistency', '1',
            '--PatchMatchStereo.max_image_size', '-1', '--PatchMatchStereo.gpu_index', '0'])
        for name in names:
            for kind in ('photometric', 'geometric'):
                for field in ('depth_maps', 'normal_maps'):
                    path = dense / 'stereo' / field / f'{name}.{kind}.bin'
                    if not path.is_file() or path.stat().st_size < 100:
                        raise RuntimeError(f'Missing PatchMatch output: {path}')
        run('stereo_fusion', ['colmap', 'stereo_fusion', '--workspace_path', str(dense),
            '--workspace_format', 'COLMAP', '--input_type', 'geometric',
            '--output_path', str(dense / 'fused.ply'), '--StereoFusion.num_threads', '8'])
        manifest['status'] = 'downsample_points'
        save()
        print('STAGE downsample_points', flush=True)
        import open3d as o3d
        cloud = o3d.io.read_point_cloud(str(dense / 'fused.ply'))
        manifest['fused_points'] = len(cloud.points)
        if len(cloud.points) < args.initial_points:
            raise ValueError('Insufficient fused points')
        cloud = cloud.farthest_point_down_sample(args.initial_points)
        if not cloud.has_colors() or not cloud.has_normals() or any(
                not np.isfinite(np.asarray(v)).all() for v in (cloud.points, cloud.colors, cloud.normals)):
            raise ValueError('Invalid initialization point attributes')
        if not o3d.io.write_point_cloud(str(points), cloud):
            raise RuntimeError('Failed to save point cloud')
        manifest.update(status='complete', initial_points=len(cloud.points),
                        initial_cloud_sha256=sha256(points))
        save()
        for dataset in (half, full):
            (dataset / 'adapter.json').write_text(json.dumps(manifest, indent=2) + '\n')
        print('PREPARATION_COMPLETE', json.dumps(manifest), flush=True)
    except Exception as error:
        manifest.update(failed_stage=manifest['status'], status='failed', error=repr(error))
        save()
        raise


if __name__ == '__main__':
    main()
