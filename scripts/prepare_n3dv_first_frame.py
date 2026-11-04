"""Reconstruct an N3DV first frame at full size and prepare half-size GS input.

Read original images through symlinks. Every invocation requires a new output
directory; existing datasets and reconstruction outputs are never overwritten.
This prepares frame zero only. Later frames must use the same new calibration.
"""

import argparse
from contextlib import contextmanager, nullcontext
from datetime import datetime
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import time

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
    parser.add_argument('--source-images', type=Path, required=True)
    parser.add_argument('--video-root', type=Path, required=True,
                        help='Verify cached frame zero against every original MP4.')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected-cameras', type=int, required=True)
    parser.add_argument('--initial-points', type=int, default=40000)
    parser.add_argument('--magick', type=Path, required=True)
    parser.add_argument('--sift-on-cpu', action='store_true',
                        help='Run feature extraction and matching on CPU; dense stereo still needs CUDA.')
    parser.add_argument('--gpu-lock', type=Path,
                        help='Share this file lock with other jobs, acquiring it only for CUDA stages.')
    args = parser.parse_args()
    images = sorted(args.source_images.resolve().glob('cam*.png'))
    if len(images) != args.expected_cameras or not any(p.name == 'cam00.png' for p in images):
        raise ValueError('Unexpected source camera list')
    if args.initial_points < 1 or not args.magick.is_file():
        raise ValueError('Invalid point budget or ImageMagick executable')
    if shutil.which('colmap') is None or shutil.which('ffmpeg') is None:
        raise RuntimeError('Activate the ReConGS environment first')
    root = args.output.absolute()
    root.mkdir(parents=True, exist_ok=False)
    source = root / 'source_frame0'
    source.mkdir()
    reconstruction = root / 'reconstruction'
    reconstruction.mkdir()
    manifest = {
        'created_local_time': datetime.now().astimezone().isoformat(),
        'scope': 'frame zero only; later frames require this same calibration',
        'source_images': str(args.source_images.resolve()),
        'video_root': str(args.video_root.resolve()),
        'cameras': [p.name for p in images], 'test_camera': 'cam00.png',
        'reconstruction_resolution': 'full', 'training_resolution': 'half',
        'source_images_copied': False, 'commands': [], 'status': 'verifying_sources',
        'sift_device': 'cpu' if args.sift_on_cpu else 'cuda',
        'gpu_lock': str(args.gpu_lock) if args.gpu_lock is not None else None,
    }

    def save():
        (root / 'preparation_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')

    @contextmanager
    def gpu_slot(stage):
        if args.gpu_lock is None:
            yield
            return
        manifest['status'] = f'waiting_for_gpu_{stage}'
        save()
        print(f'WAIT_GPU {stage}', flush=True)
        with args.gpu_lock.open('a') as lock:
            while True:
                fcntl.flock(lock, fcntl.LOCK_EX)
                try:
                    active = subprocess.check_output(['nvidia-smi',
                        '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
                        text=True).strip()
                except Exception:
                    fcntl.flock(lock, fcntl.LOCK_UN)
                    raise
                if not active:
                    break
                fcntl.flock(lock, fcntl.LOCK_UN)
                time.sleep(15)
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def run(stage, command, *, uses_gpu=False):
        with (gpu_slot(stage) if uses_gpu else nullcontext()):
            run_command(stage, command)

    def run_command(stage, command):
        manifest['status'] = stage
        manifest['commands'].append(command)
        save()
        print(f'STAGE {stage}', flush=True)
        env = dict(os.environ, QT_QPA_PLATFORM='offscreen', OMP_NUM_THREADS='8',
                   MAGICK_THREAD_LIMIT='1')
        with (reconstruction / f'{stage}.log').open('wb') as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT, env=env)
        # COLMAP worker errors can leave the parent with a zero exit status.
        errors = [line for line in (reconstruction / f'{stage}.log').read_text(
            errors='replace').splitlines() if line.startswith(('E20', 'F20'))]
        if errors:
            raise RuntimeError(f'{stage}: {errors[:3]}')

    save()
    try:
        source_audit = []
        for path in images:
            video = args.video_root.resolve() / f'{path.stem}.mp4'
            command = ['ffmpeg', '-v', 'error', '-threads', '2', '-i', str(video),
                       '-frames:v', '1', '-f', 'image2pipe', '-vcodec', 'png',
                       '-threads', '1', '-']
            decoded = subprocess.run(command, check=True, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE).stdout
            with Image.open(io.BytesIO(decoded)) as actual, Image.open(path) as cached:
                actual.load()
                cached.load()
                if (cached.mode != 'RGB' or cached.size != actual.size or
                        not np.array_equal(np.asarray(cached), np.asarray(actual))):
                    raise ValueError(f'Cached image differs from video frame zero: {path}')
                source_audit.append(dict(camera=path.name, size=list(cached.size),
                                         sha256=sha256(path), video_frame0_pixel_equal=True))
            (source / path.name).symlink_to(path)
        if len({tuple(row['size']) for row in source_audit}) != 1:
            raise ValueError('Shared-camera reconstruction requires equal source image sizes')
        manifest['source_audit'] = source_audit
        save()
        print(f'VERIFIED {len(images)} full-size video frame-zero images', flush=True)
        database = reconstruction / 'database.db'
        sparse = reconstruction / 'distorted/sparse'
        sparse.mkdir(parents=True)
        sift_gpu = '0' if args.sift_on_cpu else '1'
        run('feature_extraction', ['colmap', 'feature_extractor', '--database_path', str(database),
            '--image_path', str(source), '--ImageReader.camera_model', 'OPENCV',
            '--ImageReader.single_camera', '1', '--SiftExtraction.use_gpu', sift_gpu,
            '--SiftExtraction.max_image_size', '3200', '--SiftExtraction.num_threads', '8'],
            uses_gpu=not args.sift_on_cpu)
        run('exhaustive_match', ['colmap', 'exhaustive_matcher', '--database_path', str(database),
            '--SiftMatching.use_gpu', sift_gpu, '--SiftMatching.num_threads', '8'],
            uses_gpu=not args.sift_on_cpu)
        run('mapper', ['colmap', 'mapper', '--database_path', str(database),
            '--image_path', str(source), '--output_path', str(sparse),
            '--Mapper.multiple_models', '0', '--Mapper.num_threads', '8'])
        model = sparse / '0'
        registered = read_extrinsics_binary(model / 'images.bin')
        if sorted(im.name for im in registered.values()) != [p.name for p in images]:
            raise ValueError('Not all source cameras registered in the reconstruction')
        dense = reconstruction / 'dense'
        run('undistort_frame0', ['colmap', 'image_undistorter', '--image_path', str(source),
            '--input_path', str(model), '--output_path', str(dense), '--output_type', 'COLMAP'])
        cameras = read_intrinsics_binary(dense / 'sparse/cameras.bin')
        if len(cameras) != 1 or next(iter(cameras.values())).model != 'PINHOLE':
            raise ValueError('Expected one shared undistorted PINHOLE camera')
        camera = next(iter(cameras.values()))
        manifest['registered_cameras'] = len(registered)
        manifest['full_image_size'] = [camera.width, camera.height]
        manifest['full_camera_params'] = camera.params.tolist()
        run('patch_match', ['colmap', 'patch_match_stereo', '--workspace_path', str(dense),
            '--workspace_format', 'COLMAP', '--PatchMatchStereo.geom_consistency', '1',
            '--PatchMatchStereo.max_image_size', '-1', '--PatchMatchStereo.gpu_index', '0'],
            uses_gpu=True)
        run('stereo_fusion', ['colmap', 'stereo_fusion', '--workspace_path', str(dense),
            '--workspace_format', 'COLMAP', '--input_type', 'geometric',
            '--output_path', str(dense / 'fused.ply'), '--StereoFusion.num_threads', '8'])
        manifest['status'] = 'downsample_points'
        save()
        print('STAGE downsample_points', flush=True)
        import open3d as o3d
        pointcloud = o3d.io.read_point_cloud(str(dense / 'fused.ply'))
        manifest['fused_points'] = len(pointcloud.points)
        if len(pointcloud.points) < args.initial_points:
            raise ValueError('Fused cloud has fewer points than requested')
        pointcloud = pointcloud.farthest_point_down_sample(args.initial_points)
        points = reconstruction / 'points3D_downsample2.ply'
        if not o3d.io.write_point_cloud(str(points), pointcloud):
            raise RuntimeError('Failed to save initialization point cloud')
        for values in (pointcloud.points, pointcloud.colors, pointcloud.normals):
            if not np.isfinite(np.asarray(values)).all():
                raise ValueError('Non-finite initialization point attributes')
        if not pointcloud.has_colors() or not pointcloud.has_normals():
            raise ValueError('Initialization cloud needs colors and normals')
        half_size = [(camera.width + 1) // 2, (camera.height + 1) // 2]
        scaled = camera.params * np.array([half_size[0] / camera.width,
            half_size[1] / camera.height, half_size[0] / camera.width, half_size[1] / camera.height])
        for name in ('dataset_full_first', 'dataset_half_first'):
            dataset = root / name
            (dataset / 'frames').mkdir(parents=True)
            adapter = dataset / 'sparse/0'
            adapter.mkdir(parents=True)
            (adapter / 'images.bin').symlink_to(dense / 'sparse/images.bin')
            (dataset / 'points3D.ply').symlink_to(points)
            (adapter / 'points3D.ply').symlink_to('../../points3D.ply')
            (dataset / 'test_split.txt').write_text('cam00.png\n')
            if name == 'dataset_full_first':
                (dataset / 'frames/0000').symlink_to(dense / 'images', target_is_directory=True)
                (adapter / 'cameras.bin').symlink_to(dense / 'sparse/cameras.bin')
            else:
                frame = dataset / 'frames/0000'
                frame.mkdir()
                for path in images:
                    run(f'resize_{path.stem}', [str(args.magick), str(dense / 'images' / path.name),
                        '-resize', '50%', str(frame / path.name)])
                    with Image.open(frame / path.name) as image:
                        image.load()
                        if list(image.size) != half_size or image.mode != 'RGB':
                            raise ValueError(f'Invalid half-size image: {path.name}')
                with (adapter / 'cameras.bin').open('wb') as output:
                    output.write(struct.pack('<QiiQQdddd', 1, camera.id, 1, *half_size, *scaled))
        manifest.update(status='complete', half_image_size=half_size,
            half_camera_params=scaled.tolist(), initial_points=len(pointcloud.points),
            initial_cloud_sha256=sha256(points),
            undistorted_camera_sha256=sha256(dense / 'sparse/cameras.bin'),
            undistorted_extrinsics_sha256=sha256(dense / 'sparse/images.bin'),
            adapter_note='Half-size images.bin retains full-size 2D tracks; it is a loader adapter, not a rescaled SfM model.')
        save()
        print('PREPARATION_COMPLETE', json.dumps({key: manifest[key] for key in
            ('registered_cameras', 'full_image_size', 'half_image_size', 'fused_points', 'initial_points')}), flush=True)
    except Exception as error:
        manifest['failed_stage'] = manifest['status']
        manifest['status'] = 'failed'
        manifest['error'] = repr(error)
        save()
        raise


if __name__ == '__main__':
    main()
