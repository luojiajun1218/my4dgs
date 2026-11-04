"""Prepare consecutive half-size frames with an existing first-frame calibration.

Decode original videos into a private temporary workspace, reuse the exact
COLMAP undistortion and ImageMagick resize used for frame zero, and link all
shared calibration/point-cloud files. No reconstruction or training is run.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

from PIL import Image


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preparation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--num-frames', type=int, default=30)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--decode-batch-size', type=int,
        help='Decode and convert this many frames at a time to bound temporary disk use.')
    parser.add_argument('--link', type=Path)
    parser.add_argument('--magick', type=Path,
        default=Path('/home/gamer-02/Documents/Codex/miniforge3/envs/s2gs/bin/magick'))
    args = parser.parse_args()
    if not 2 <= args.num_frames <= 300 or not 1 <= args.workers <= 4:
        parser.error('Expected 2..300 frames and 1..4 workers')
    if args.decode_batch_size is not None and not 1 <= args.decode_batch_size <= 300:
        parser.error('decode-batch-size must be in 1..300')
    if any(shutil.which(command) is None for command in ('ffmpeg', 'colmap')):
        raise RuntimeError('Activate ReConGS first')
    if not args.magick.is_file():
        raise FileNotFoundError(args.magick)
    if args.link is not None and os.path.lexists(args.link):
        raise FileExistsError(args.link)
    preparation = args.preparation.resolve()
    original = json.loads((preparation / 'preparation_manifest.json').read_text())
    if original['status'] != 'complete':
        raise ValueError('First-frame preparation must be complete')
    names = original['cameras']
    first = preparation / 'dataset_half_first'
    model = preparation / 'reconstruction/distorted/sparse/0'
    reference = preparation / 'reconstruction/dense/sparse'
    expected_hashes = {
        'cameras.bin': original['undistorted_camera_sha256'],
        'images.bin': original['undistorted_extrinsics_sha256'],
    }
    if any(sha256(reference / name) != value for name, value in expected_hashes.items()):
        raise ValueError('First-frame calibration has changed')
    if sha256(first / 'points3D.ply') != original['initial_cloud_sha256']:
        raise ValueError('First-frame initialization cloud has changed')
    size = tuple(original['half_image_size'])
    source_size = tuple(original['source_audit'][0]['size'])
    root = args.output.absolute()
    root.mkdir(parents=True, exist_ok=False)
    logs, work = root / 'logs', root / 'work'
    logs.mkdir()
    work.mkdir()
    raw = work / 'raw'
    videos = root / 'source_videos'
    videos.mkdir()
    for name in names:
        video = Path(original['video_root']) / f'{Path(name).stem}.mp4'
        if not video.is_file():
            raise FileNotFoundError(video)
        (videos / video.name).symlink_to(video)
    staging = root / 'dataset.staging'
    frames = staging / 'frames'
    frames.mkdir(parents=True)
    (frames / '0000').symlink_to(first / 'frames/0000', target_is_directory=True)
    sparse = staging / 'sparse/0'
    sparse.mkdir(parents=True)
    for name in ('cameras.bin', 'images.bin'):
        (sparse / name).symlink_to(first / 'sparse/0' / name)
    (staging / 'points3D.ply').symlink_to(first / 'points3D.ply')
    (sparse / 'points3D.ply').symlink_to('../../points3D.ply')
    (staging / 'test_split.txt').symlink_to(first / 'test_split.txt')
    manifest = dict(created_local_time=datetime.now().astimezone().isoformat(),
        first_frame_preparation=str(preparation), frame_count=args.num_frames,
        cameras=names, test_camera='cam00.png', half_image_size=list(size),
        calibration_model=str(model), undistorted_model_sha256=expected_hashes,
        shared_initial_cloud_sha256=original['initial_cloud_sha256'],
        frame0_images_linked=True, source_videos_copied=False,
        temporal_shift_applied=False, source_frame_indices=list(range(args.num_frames)),
        decode_batch_size=args.decode_batch_size or args.num_frames - 1,
        status='decode')

    def save():
        (root / 'preparation_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')

    env = dict(os.environ, QT_QPA_PLATFORM='offscreen', OMP_NUM_THREADS='4',
               MAGICK_THREAD_LIMIT='1')

    def run(command, log):
        with log.open('wb') as stream:
            subprocess.run(command, check=True, env=env,
                           stdout=stream, stderr=subprocess.STDOUT)
        errors = [line for line in log.read_text(errors='replace').splitlines()
                  if line.startswith(('E20', 'F20'))]
        if errors:
            raise RuntimeError(f'{log.name}: {errors[:3]}')

    def decode(name, start, end):
        camera = Path(name).stem
        run(['ffmpeg', '-nostdin', '-v', 'error', '-threads', '2',
             '-i', str(videos / f'{camera}.mp4'), '-vf',
             f'trim=start_frame={start}:end_frame={end},setpts=PTS-STARTPTS',
             '-fps_mode', 'passthrough', '-frames:v', str(end - start),
             '-start_number', str(start), '-compression_level', '1', '-threads', '1',
             str(raw / '%04d' / name)], logs / f'decode_{camera}_{start:04d}_{end - 1:04d}.log')
        for index in range(start, end):
            with Image.open(raw / f'{index:04d}' / name) as image:
                if image.size != source_size or image.mode != 'RGB':
                    raise ValueError(f'Invalid decoded frame: {index} {name}')
        return camera

    def convert(index):
        tag = f'{index:04d}'
        temporary = work / f'undistort_{tag}'
        run(['colmap', 'image_undistorter', '--image_path', str(raw / tag),
             '--input_path', str(model), '--output_path', str(temporary),
             '--output_type', 'COLMAP'], logs / f'undistort_{tag}.log')
        for name, expected in expected_hashes.items():
            if sha256(temporary / 'sparse' / name) != expected:
                raise ValueError(f'Frame {tag} calibration differs: {name}')
        target = frames / tag
        target.mkdir()
        for name in names:
            run([str(args.magick), str(temporary / 'images' / name),
                 '-resize', '50%', str(target / name)],
                logs / f'resize_{tag}_{Path(name).stem}.log')
            with Image.open(target / name) as image:
                image.load()
                if image.size != size or image.mode != 'RGB':
                    raise ValueError(f'Invalid processed frame: {tag} {name}')
        shutil.rmtree(temporary)
        shutil.rmtree(raw / tag)
        return tag

    save()
    try:
        checks = []
        import io
        import numpy as np
        batch_size = manifest['decode_batch_size']
        for start in range(1, args.num_frames, batch_size):
            end = min(start + batch_size, args.num_frames)
            for index in range(start, end):
                (raw / f'{index:04d}').mkdir(parents=True)
            manifest.update(status='decode', active_batch=[start, end - 1])
            save()
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                for count, future in enumerate(as_completed([
                        pool.submit(decode, name, start, end) for name in names]), 1):
                    camera = future.result()
                    print(f'DECODE {count}/{len(names)} {camera} frames={start}..{end - 1}', flush=True)
            # Verify both boundaries of each batch against independent decoding.
            for index in sorted({start, end - 1}):
                decoded = subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-threads', '2',
                    '-i', str(videos / 'cam00.mp4'), '-vf', f'select=eq(n\\,{index})',
                    '-frames:v', '1', '-fps_mode', 'passthrough', '-f', 'image2pipe',
                    '-vcodec', 'png', '-threads', '1', '-'], check=True,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout
                with Image.open(io.BytesIO(decoded)) as actual, Image.open(raw / f'{index:04d}' / 'cam00.png') as batch:
                    if not np.array_equal(np.asarray(actual), np.asarray(batch)):
                        raise ValueError(f'Video frame numbering mismatch: {index}')
                checks.append(index)
            manifest.update(status='undistort', cam00_independent_decode_checks=checks)
            save()
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                for future in as_completed([pool.submit(convert, index)
                        for index in range(start, end)]):
                    tag = future.result()
                    print(f'UNDISTORT frame={tag}/{args.num_frames - 1:04d}', flush=True)
            manifest['processed_frames'] = end
            save()
        expected_names = set(names)
        for index in range(args.num_frames):
            if {p.name for p in (frames / f'{index:04d}').iterdir()} != expected_names:
                raise ValueError(f'Missing or extra cameras at frame {index}')
        staging.rename(root / 'dataset')
        shutil.rmtree(work)
        manifest.update(status='complete', image_count=args.num_frames * len(names),
                        all_new_images_decoded_and_validated=True,
                        all_frame_models_equal_first_frame=True,
                        temporary_raw_frames_removed=True)
        save()
        if args.link is not None:
            args.link.absolute().symlink_to(root / 'dataset', target_is_directory=True)
        print('SEQUENCE_PREPARATION_COMPLETE', json.dumps(dict(
            frames=args.num_frames, cameras=len(names), image_size=size,
            dataset=str(root / 'dataset'))), flush=True)
    except Exception as error:
        manifest.update(failed_stage=manifest['status'], status='failed', error=repr(error))
        save()
        raise


if __name__ == '__main__':
    main()
