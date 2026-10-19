"""Independently re-extract raw frame indices and check processed RGB pixels."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np
from PIL import Image

from prepare_n3dv_from_videos import read_png


def pixel_hash(path):
    with Image.open(path) as image:
        return hashlib.sha256(np.asarray(image.convert('RGB')).tobytes()).hexdigest()


def expected_frames(source, magick):
    manifest = json.loads((source / 'preprocessing_manifest.json').read_text())
    cameras = [manifest['camera_names'][0], manifest['camera_names'][-1]]
    frames = (0, 149, 299)
    expected = {}
    model = source / 'reconstruction/distorted/sparse/0'
    with tempfile.TemporaryDirectory(prefix='alignment_') as temporary:
        temporary = Path(temporary)
        names = temporary / 'images.txt'
        names.write_text('\n'.join(cameras) + '\n')
        for frame in frames:
            (temporary / str(frame) / 'raw').mkdir(parents=True)
        for camera in cameras:
            video = Path(manifest['raw']) / (Path(camera).stem + '.mp4')
            command = [shutil.which('ffmpeg'), '-v', 'error', '-nostdin', '-threads', '2', '-i', str(video),
                       '-vf', r'select=eq(n\,0)+eq(n\,149)+eq(n\,299)', '-fps_mode', 'passthrough',
                       '-frames:v', '3', '-c:v', 'png', '-pix_fmt', 'rgb24', '-threads', '1',
                       '-f', 'image2pipe', 'pipe:1']
            with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as process:
                for frame in frames:
                    (temporary / str(frame) / 'raw' / camera).write_bytes(read_png(process.stdout))
                extra, error = process.communicate(timeout=60)
                if process.returncode != 0 or extra:
                    raise ValueError(f'Independent video decode failed: {error!r}')
            if pixel_hash(temporary / '0/raw' / camera) != pixel_hash(source / 'reconstruction/raw_images' / camera):
                raise ValueError('SfM first image does not match raw video frame zero')
        for frame in frames:
            working = temporary / str(frame)
            subprocess.run([shutil.which('colmap'), 'image_undistorter', '--image_path', str(working / 'raw'),
                            '--input_path', str(model), '--image_list_path', str(names),
                            '--output_path', str(working / 'undistorted'), '--output_type', 'COLMAP'],
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            for camera in cameras:
                target = working / camera
                subprocess.run([str(magick), str(working / 'undistorted/images' / camera),
                                '-resize', '50%', '-type', 'TrueColor', str(target)],
                               check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                expected[f'{Path(camera).stem}/images/{frame:04d}.png'] = pixel_hash(target)
    return dict(scene=source.name, sampled_raw_frames=list(frames), sampled_cameras=cameras,
                independent_select_filter=True, sfm_first_images_match_raw_frame_zero=True,
                expected_rgb_sha256=expected,
                model_sha256={name: hashlib.sha256((model / name).read_bytes()).hexdigest()
                              for name in ('cameras.bin', 'images.bin', 'points3D.bin')})


def verify_prepared(source, expected):
    model = source / 'reconstruction/distorted/sparse/0'
    for name, digest in expected['model_sha256'].items():
        if hashlib.sha256((model / name).read_bytes()).hexdigest() != digest:
            raise ValueError('Calibration changed since independent frame extraction')
    for name, digest in expected['expected_rgb_sha256'].items():
        if pixel_hash(source / name) != digest:
            raise ValueError('Raw/processed camera or frame index mismatch: ' + name)
    return dict(expected, status='passed', processed_pixels_match_independent_reextraction=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expectation', type=Path)
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--magick', type=Path, default=Path('/home/gamer-02/Documents/Codex/miniforge3/envs/s2gs/bin/magick'))
    args = parser.parse_args()
    result = (json.loads(args.expectation.read_text()) if args.expectation
              else expected_frames(args.source, args.magick))
    if args.verify:
        result = verify_prepared(args.source, result)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
