"""Rebuild N3DV with one full-resolution COLMAP/MVS model per scene.

Raw videos stay in place. Only generated first-frame reconstruction artifacts
and half-size undistorted training images are retained. Subsequent full-size
frames are streamed through a temporary workspace, without duplicating a video
or holding a second complete image sequence on disk.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from datetime import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import time

import numpy as np
from PIL import Image
from plyfile import PlyData
from scipy.spatial import cKDTree

from libgs.data.utils.colmap import (
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary,
)


def write_json(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as file:
        for block in iter(lambda: file.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_exact(file, count):
    parts = []
    while count:
        block = file.read(count)
        if not block:
            raise EOFError('Video decoder ended before the requested frame')
        parts.append(block)
        count -= len(block)
    return b''.join(parts)


def read_png(file):
    """Read exactly one PNG from image2pipe; never shift a missing frame."""
    signature = read_exact(file, 8)
    if signature != b'\x89PNG\r\n\x1a\n':
        raise ValueError('Unexpected image2pipe signature')
    pieces = [signature]
    while True:
        header = read_exact(file, 8)
        length = struct.unpack('>I', header[:4])[0]
        if length > 128 * 1024 * 1024:
            raise ValueError('Unexpectedly large PNG chunk')
        pieces.extend((header, read_exact(file, length + 4)))
        if header[4:] == b'IEND':
            return b''.join(pieces)


def move_flat_model(workspace):
    sparse = workspace / 'sparse'
    target = sparse / '0'
    target.mkdir(exist_ok=True)
    for name in ('cameras.bin', 'images.bin', 'points3D.bin'):
        source = sparse / name
        if source.exists():
            source.replace(target / name)
        if not (target / name).is_file():
            raise FileNotFoundError(target / name)
    return target


def seed_geometry(scene, model, seed_path, fused_path):
    intrinsics = read_intrinsics_binary(model / 'cameras.bin')
    extrinsics = read_extrinsics_binary(model / 'images.bin')
    seed_data = PlyData.read(seed_path)['vertex'].data
    fused_data = PlyData.read(fused_path)['vertex'].data
    xyz = np.column_stack([seed_data[k] for k in ('x', 'y', 'z')]).astype(np.float64)
    fused_xyz = np.column_stack([fused_data[k] for k in ('x', 'y', 'z')]).astype(np.float64)
    if len(xyz) != min(40000, len(fused_xyz)) or not np.isfinite(xyz).all():
        raise ValueError('Invalid downsampled seed point cloud')
    distances, _ = cKDTree(fused_xyz).query(xyz, workers=4)
    if np.max(distances) != 0:
        raise ValueError('Seed is not a subset of the matching fused cloud')
    centers = []
    projected_counts = np.zeros(len(xyz), dtype=np.int32)
    cameras = []
    for extr in sorted(extrinsics.values(), key=lambda item: item.name):
        intr = intrinsics[extr.camera_id]
        rotation = extr.qvec2rotmat()
        if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-8):
            raise ValueError('Invalid camera rotation')
        center = -rotation.T @ extr.tvec
        if extr.name != 'cam00.png':
            centers.append(center)
        position = xyz @ rotation.T + extr.tvec
        positive = position[:, 2] > 0
        uv = position[:, :2] / np.maximum(position[:, 2:], 1e-12)
        fx, fy, cx, cy = intr.params
        uv = uv * [fx, fy] + [cx, cy]
        inside = positive & (uv[:, 0] >= 0) & (uv[:, 0] < intr.width)
        inside &= (uv[:, 1] >= 0) & (uv[:, 1] < intr.height)
        projected_counts += inside
        cameras.append(dict(name=extr.name, full_size=[intr.width, intr.height],
                            fx=float(fx), fy=float(fy), cx=float(cx), cy=float(cy),
                            seed_points_inside=int(inside.sum())))
    centers = np.asarray(centers)
    extent = float(1.1 * np.linalg.norm(centers - centers.mean(0), axis=1).max())
    radius = np.linalg.norm(xyz - np.median(xyz, axis=0), axis=1)
    return dict(seed_points=len(xyz), fused_points=len(fused_xyz), finite=True,
                seed_to_fused_max_distance=float(distances.max()),
                camera_extent=extent, radius_quantiles=np.quantile(radius, [.5, .99, .999, 1]).tolist(),
                far_over_10_camera_extents=int((radius > 10 * extent).sum()),
                projected_inside_fewer_than_two_cameras=int((projected_counts < 2).sum()),
                distance_is_diagnostic_not_a_deletion_rule=True, cameras=cameras)


def prepare_scene(raw, destination, args):
    resume = getattr(args, 'resume', False)
    if destination.is_symlink() or (destination.exists() and not resume):
        raise FileExistsError(f'Refuse to overwrite an existing scene: {destination}')
    videos = sorted(raw.glob('cam*.mp4'))
    if len(videos) < 3 or videos[0].stem != 'cam00':
        raise ValueError(f'Missing camera videos, including cam00: {raw}')
    existing = destination.exists()
    if existing:
        manifest = json.loads((destination / 'preprocessing_manifest.json').read_text())
        if (manifest['raw'] != str(raw.resolve()) or manifest['scene'] != raw.name or
                manifest['frames'] != args.frames or
                manifest['camera_names'] != [video.stem + '.png' for video in videos]):
            raise ValueError('Resume inputs differ from the original scene manifest')
        if manifest['status'] == 'complete':
            hashes = manifest.get('metadata_sha256', {})
            if not hashes or any(sha256(destination / name) != digest for name, digest in hashes.items()):
                raise ValueError('Completed scene metadata changed; do not silently reuse it')
            expected = [f'{frame:04d}.png' for frame in range(args.frames)]
            for video in videos:
                images = destination / video.stem / 'images'
                if sorted(path.name for path in images.glob('*.png')) != expected:
                    raise ValueError('Completed scene has missing or extra frame images')
            print(json.dumps(dict(scene=raw.name, stage='already_complete', reused_verified_data=True)), flush=True)
            return
        manifest.setdefault('resume_history', []).append(dict(
            resumed_at=datetime.now().astimezone().isoformat(),
            previous_stage=manifest.get('stage'), previous_error=manifest.pop('error', None)))
        manifest['status'] = 'preparing'
    else:
        manifest = dict(status='preparing', scene=raw.name, raw=str(raw.resolve()),
                        destination=str(destination), started_at=datetime.now().astimezone().isoformat(),
                        frames=args.frames, camera_names=[v.stem + '.png' for v in videos],
                        test_camera='cam00.png', commands=[], source_videos_copied=False)
    destination.mkdir(parents=True, exist_ok=True)
    if not (destination / 'video_source').is_symlink():
        (destination / 'video_source').symlink_to(raw.resolve(), target_is_directory=True)
    logs = destination / 'logs'
    logs.mkdir(exist_ok=True)
    reconstruction = destination / 'reconstruction'
    inputs = reconstruction / 'raw_images'
    inputs.mkdir(parents=True, exist_ok=True)
    completed_stages = {record['stage'] for record in manifest['commands']}
    manifest_path = destination / 'preprocessing_manifest.json'
    write_json(manifest_path, manifest)

    def update(stage, **fields):
        manifest.update(stage=stage, **fields)
        write_json(manifest_path, manifest)
        print(json.dumps(dict(scene=raw.name, stage=stage, **fields), ensure_ascii=False), flush=True)

    def run(stage, command):
        if existing and stage in completed_stages:
            print(json.dumps(dict(scene=raw.name, stage=stage, reuse_completed_stage=True)), flush=True)
            return
        start = time.perf_counter()
        update(stage)
        with (logs / (stage + '.log')).open('w') as log:
            subprocess.run([str(x) for x in command], check=True, stdout=log, stderr=subprocess.STDOUT,
                           env=dict(os.environ, QT_QPA_PLATFORM='offscreen'))
        manifest['commands'].append(dict(stage=stage, argv=[str(x) for x in command],
                                         seconds=time.perf_counter() - start))
        write_json(manifest_path, manifest)

    processes = []
    try:
        with ExitStack() as stack:
            for video in videos:
                stderr = stack.enter_context((logs / (video.stem + '_decode.log')).open('wb'))
                command = [args.ffmpeg, '-v', 'error', '-nostdin', '-threads', '2', '-i', str(video),
                           '-map', '0:v:0', '-frames:v', str(args.frames), '-vf', 'scale=iw:ih',
                           '-fps_mode', 'passthrough', '-c:v', 'png', '-pix_fmt', 'rgb24',
                           '-compression_level', '1', '-threads', '1', '-f', 'image2pipe', 'pipe:1']
                process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=stderr)
                processes.append((video, process))
                first_png = read_png(process.stdout)
                original = inputs / (video.stem + '.png')
                if original.exists():
                    with Image.open(original) as before, Image.open(io.BytesIO(first_png)) as after:
                        if not np.array_equal(np.asarray(before), np.asarray(after)):
                            raise ValueError('Raw first frame changed since camera reconstruction')
                else:
                    if existing and 'mapping' in completed_stages:
                        raise FileNotFoundError('Cannot resume a reconstructed scene with missing raw first images')
                    original.write_bytes(first_png)
            update('first_frames_decoded', cameras=len(videos))
            database = reconstruction / 'database.db'
            distorted = reconstruction / 'distorted/sparse'
            distorted.mkdir(parents=True, exist_ok=True)
            run('feature_extraction', [args.colmap, 'feature_extractor', '--database_path', database,
                '--image_path', inputs, '--ImageReader.single_camera', '1', '--ImageReader.camera_model', 'OPENCV',
                '--SiftExtraction.use_gpu', '1', '--SiftExtraction.num_threads', str(args.threads)])
            run('matching', [args.colmap, 'exhaustive_matcher', '--database_path', database,
                '--SiftMatching.use_gpu', '1', '--SiftMatching.num_threads', str(args.threads)])
            run('mapping', [args.colmap, 'mapper', '--database_path', database, '--image_path', inputs,
                '--output_path', distorted, '--Mapper.ba_global_function_tolerance', '0.000001',
                '--Mapper.num_threads', str(args.threads), '--random_seed', '0'])
            source_model = distorted / '0'
            expected_names = [v.stem + '.png' for v in videos]
            registered = read_extrinsics_binary(source_model / 'images.bin')
            if sorted(v.name for v in registered.values()) != expected_names:
                raise ValueError('COLMAP did not register every source camera; do not drop cameras silently')
            first = destination / 'frame000001'
            run('first_undistortion', [args.colmap, 'image_undistorter', '--image_path', inputs,
                '--input_path', source_model, '--output_path', first, '--output_type', 'COLMAP'])
            # COLMAP's dense workspace expects sparse/*.bin, not sparse/0.
            model = first / 'sparse'
            if not (model / 'cameras.bin').exists() and (model / '0/cameras.bin').exists():
                model = model / '0'
            intrinsics = read_intrinsics_binary(model / 'cameras.bin')
            for intr in intrinsics.values():
                if intr.model != 'PINHOLE' or not np.allclose(intr.params[2:], [intr.width / 2, intr.height / 2], atol=1e-7):
                    raise ValueError('Training requires undistorted PINHOLE cameras with centered principal points')
            if existing and 'dense_stereo' not in completed_stages:
                if model.name == '0':
                    raise ValueError('Pending dense stereo requires the original flat COLMAP workspace')
                # A CUDA failure may leave apparently present but incomplete maps.
                # Recompute these caches while preserving both camera models.
                preserved_model_hashes = {name: sha256(model / name)
                                          for name in ('cameras.bin', 'images.bin', 'points3D.bin')}
                for name in ('depth_maps', 'normal_maps', 'consistency_graphs'):
                    cache = first / 'stereo' / name
                    if cache.is_symlink():
                        raise ValueError('Refuse to clear a symlinked stereo cache')
                    if cache.exists():
                        shutil.rmtree(cache)
                    cache.mkdir(parents=True)
                if any(sha256(model / name) != digest for name, digest in preserved_model_hashes.items()):
                    raise ValueError('Camera reconstruction changed while clearing failed caches')
                manifest['resume_history'][-1]['preserved_undistorted_model_sha256'] = preserved_model_hashes
            run('dense_stereo', [args.colmap, 'patch_match_stereo', '--workspace_path', first,
                '--workspace_format', 'COLMAP', '--PatchMatchStereo.geom_consistency', 'true'])
            fused = first / 'fused.ply'
            run('fusion', [args.colmap, 'stereo_fusion', '--workspace_path', first,
                '--workspace_format', 'COLMAP', '--input_type', 'geometric', '--output_path', fused,
                '--StereoFusion.num_threads', str(args.threads)])
            model = move_flat_model(first)
            run('sample_seed', [args.python, args.s2gs / 'scripts/pre_dynerf/downsample_point.py',
                                fused, destination / 'points3D_downsample2.ply'])
            geometry = seed_geometry(destination, model, destination / 'points3D_downsample2.ply', fused)
            write_json(destination / 'seed_geometry.json', geometry)
            run('llff_metadata', [args.python, args.s2gs / 'scripts/pre_dynerf/imgs2poses.py',
                '--scenedir', first, '--outdir', destination])

            def resize(item):
                name, source, frame = item
                target = destination / Path(name).stem / 'images' / f'{frame:04d}.png'
                target.parent.mkdir(parents=True, exist_ok=True)
                subprocess.run([str(args.magick), str(source), '-resize', '50%', '-type', 'TrueColor',
                                '-define', 'png:compression-level=3', str(target)],
                               check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

            with ThreadPoolExecutor(max_workers=4) as pool:
                list(pool.map(resize, [(name, first / 'images' / name, 0) for name in expected_names]))
                # Each subsequent frame uses the same ORIGINAL distorted model.
                # The temporary full-size images never accumulate into another dataset.
                for frame in range(1, args.frames):
                    with tempfile.TemporaryDirectory(prefix='frame_', dir=destination) as temporary:
                        temporary = Path(temporary)
                        raw_images = temporary / 'raw'
                        raw_images.mkdir()
                        for video, process in processes:
                            (raw_images / (video.stem + '.png')).write_bytes(read_png(process.stdout))
                        undistorted = temporary / 'undistorted'
                        with (logs / 'sequence_undistortion.log').open('a') as log:
                            subprocess.run([str(args.colmap), 'image_undistorter', '--image_path', str(raw_images),
                                '--input_path', str(source_model), '--output_path', str(undistorted),
                                '--output_type', 'COLMAP'], check=True, stdout=log, stderr=subprocess.STDOUT)
                        list(pool.map(resize, [(name, undistorted / 'images' / name, frame) for name in expected_names]))
                    if (frame + 1) % 30 == 0 or frame == args.frames - 1:
                        update('sequence_images', completed_frames=frame + 1)
            for video, process in processes:
                if process.stdout.read(1) or process.wait(timeout=30) != 0:
                    raise ValueError(f'Unexpected video decoder result: {video}')
                process.stdout.close()

        update('validating')
        extrinsics = read_extrinsics_binary(model / 'images.bin')
        image_count = 0
        sizes = {}
        for extr in extrinsics.values():
            intr = intrinsics[extr.camera_id]
            expected_size = ((intr.width + 1) // 2, (intr.height + 1) // 2)
            directory = destination / Path(extr.name).stem / 'images'
            if sorted(p.name for p in directory.iterdir()) != [f'{f:04d}.png' for f in range(args.frames)]:
                raise ValueError('Incomplete frame sequence: ' + str(directory))
            for path in sorted(directory.iterdir()):
                with Image.open(path) as image:
                    if image.size != expected_size or image.mode != 'RGB':
                        raise ValueError('Invalid training image size/mode: ' + str(path))
                    image.verify()
                image_count += 1
            for frame in (0, args.frames // 2, args.frames - 1):
                with Image.open(directory / f'{frame:04d}.png') as image:
                    image.load()
            sizes[Path(extr.name).stem] = list(expected_size)
        metadata_hashes = {str(p.relative_to(destination)): sha256(p) for p in (
            model / 'cameras.bin', model / 'images.bin', model / 'points3D.bin',
            source_model / 'cameras.bin', source_model / 'images.bin', source_model / 'points3D.bin',
            fused, destination / 'points3D_downsample2.ply', destination / 'poses_bounds.npy')}
        update('complete', status='complete', validated_pngs=image_count, half_image_sizes=sizes,
               geometry=geometry, metadata_sha256=metadata_hashes,
               finished_at=datetime.now().astimezone().isoformat())
        # Keep inputs, database, BOTH calibration models and fused/seed PLYs.
        # Depth/normal caches can be regenerated and are not training inputs.
        shutil.rmtree(first / 'stereo')
    except BaseException as error:
        update('failed', status='failed', error=repr(error))
        raise
    finally:
        for _, process in processes:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-root', type=Path, required=True)
    parser.add_argument('--destination-root', type=Path, required=True)
    parser.add_argument('--scenes', nargs='+', required=True)
    parser.add_argument('--frames', type=int, default=300)
    parser.add_argument('--threads', type=int, default=16)
    parser.add_argument('--resume', action='store_true',
                        help='Resume a matching generated scene; preserve completed calibration and discard failed stereo caches')
    parser.add_argument('--colmap', type=Path, default=Path(shutil.which('colmap') or 'colmap'))
    parser.add_argument('--ffmpeg', type=Path, default=Path(shutil.which('ffmpeg') or 'ffmpeg'))
    parser.add_argument('--magick', type=Path, default=Path('/home/gamer-02/Documents/Codex/miniforge3/envs/s2gs/bin/magick'))
    parser.add_argument('--python', type=Path, default=Path(os.sys.executable))
    parser.add_argument('--s2gs', type=Path, default=Path('/home/gamer-02/Documents/DeepLearning/S2GS'))
    args = parser.parse_args()
    if args.frames < 1:
        parser.error('frames must be positive')
    pending = False
    for scene in args.scenes:
        marker = args.destination_root / scene / 'preprocessing_manifest.json'
        if not args.resume or not marker.exists() or json.loads(marker.read_text()).get('status') != 'complete':
            pending = True
    if pending:
        gpu = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                             capture_output=True, text=True, timeout=15)
        if gpu.returncode or not gpu.stdout.strip():
            raise RuntimeError('GPU unavailable; no preprocessing was started. Restore the GPU before resuming.')
    for scene in args.scenes:
        prepare_scene(args.raw_root / scene, args.destination_root / scene, args)


if __name__ == '__main__':
    main()
