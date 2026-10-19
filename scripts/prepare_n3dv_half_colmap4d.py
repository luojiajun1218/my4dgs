"""Link existing half-size N3DV images and scale retained COLMAP intrinsics.

This creates a loader adapter, not a new SfM reconstruction. A retained
undistorted PINHOLE model is required; LLFF's single focal is insufficient to
recover the original fx/fy. Shared images, poses and seed PLY stay untouched.
"""

import argparse
import hashlib
import json
from pathlib import Path
import struct

import numpy as np
from PIL import Image

from libgs.data.utils.colmap import read_extrinsics_binary, read_intrinsics_binary


def prepare(source: Path, destination: Path, frames: int):
    if frames < 1:
        raise ValueError('frames must be positive')
    source = source.resolve(strict=True)
    # Preserve the requested destination rather than resolving an existing link.
    destination = destination.absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    model = source / 'frame000001/sparse/0'
    intrinsics = read_intrinsics_binary(model / 'cameras.bin')
    extrinsics = read_extrinsics_binary(model / 'images.bin')
    names = sorted(extr.name for extr in extrinsics.values())
    if len(names) != len(set(names)) or 'cam00.png' not in names:
        raise ValueError('Expected unique camera names, including cam00.png')
    image_cameras = sorted(path.name for path in source.glob('cam*') if path.is_dir())
    if image_cameras != [Path(name).stem for name in names]:
        raise ValueError('COLMAP cameras differ from existing image directories')
    scaled = {}
    links = []
    for extr in extrinsics.values():
        name = Path(extr.name)
        if name.name != extr.name or name.suffix != '.png':
            raise ValueError('Expected bare PNG camera filenames')
        intr = intrinsics[extr.camera_id]
        if intr.model != 'PINHOLE':
            raise ValueError('Retained undistorted PINHOLE cameras are required')
        dimensions = None
        for frame in range(frames):
            target = source / name.stem / 'images' / f'{frame:04d}.png'
            with Image.open(target) as image:
                if image.mode != 'RGB':
                    raise ValueError(f'Expected RGB image: {target}')
                size = image.size
            if size != ((intr.width + 1) // 2, (intr.height + 1) // 2):
                raise ValueError(f'Image is not half the calibrated dimensions: {target}')
            if dimensions is not None and dimensions != size:
                raise ValueError(f'Image dimensions change across frames: {target}')
            dimensions = size
            links.append((frame, extr.name, target))
        width, height = dimensions
        sx, sy = width / intr.width, height / intr.height
        fx, fy, cx, cy = intr.params
        if not np.allclose([cx, cy], [intr.width / 2, intr.height / 2], atol=1e-7):
            raise ValueError('libgs assumes a centered principal point')
        camera = dict(id=intr.id, width=width, height=height,
                      params=[fx * sx, fy * sy, cx * sx, cy * sy])
        if intr.id in scaled and scaled[intr.id] != camera:
            raise ValueError('Shared camera has inconsistent dimensions')
        scaled[intr.id] = camera
    ply = source / 'points3D_downsample2.ply'
    if not ply.is_file():
        raise FileNotFoundError(ply)
    destination.mkdir(parents=True)
    sparse = destination / 'sparse/0'
    sparse.mkdir(parents=True)
    with (sparse / 'cameras.bin').open('xb') as file:
        file.write(struct.pack('<Q', len(scaled)))
        for camera in sorted(scaled.values(), key=lambda item: item['id']):
            file.write(struct.pack('<iiQQ', camera['id'], 1, camera['width'], camera['height']))
            file.write(struct.pack('<dddd', *camera['params']))
    (sparse / 'images.bin').symlink_to(model / 'images.bin')
    (sparse / 'points3D.ply').symlink_to(ply)
    for frame, name, target in links:
        directory = destination / 'frames' / f'{frame:04d}'
        directory.mkdir(parents=True, exist_ok=True)
        (directory / name).symlink_to(target)
    (destination / 'test_split.txt').write_text('cam00.png\n')
    metadata = dict(source=str(source), frames=frames, camera_names=names,
                    image_links=len(links), scaled_cameras=list(scaled.values()),
                    seed_sha256=hashlib.sha256(ply.read_bytes()).hexdigest(),
                    source_intrinsics_sha256=hashlib.sha256((model / 'cameras.bin').read_bytes()).hexdigest(),
                    note='Loader adapter only. images.bin 2D observations retain original pixel coordinates; do not use this adapter as an SfM model.')
    (destination / 'adapter.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--destination', required=True, type=Path)
    parser.add_argument('--frames', type=int, default=300)
    args = parser.parse_args()
    prepare(args.source, args.destination, args.frames)
