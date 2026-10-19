"""Check regenerated N3DV data through the actual native/dynamic loader."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
from plyfile import PlyData
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from libgs.data import load_scene
from libgs.data.utils.colmap import read_extrinsics_binary, read_intrinsics_binary
from pipeline.data import DataConfig, DataModule
from prepare_n3dv_half_colmap4d import prepare


def validate(source, adapter, create=False):
    source = source.resolve(strict=True)
    manifest = json.loads((source / 'preprocessing_manifest.json').read_text())
    if manifest['status'] != 'complete' or manifest['frames'] != 300:
        raise ValueError('A complete, validated 300-frame scene is required')
    if create and not adapter.exists():
        prepare(source, adapter, frames=300)
    metadata = json.loads((adapter / 'adapter.json').read_text())
    if Path(metadata['source']).resolve() != source or metadata['frames'] != 300:
        raise ValueError('Adapter refers to a different scene or frame count')
    for name, digest in manifest['metadata_sha256'].items():
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != digest:
            raise ValueError('Regenerated input changed: ' + name)
    if not (adapter / 'sparse/0/points3D.ply').samefile(source / 'points3D_downsample2.ply'):
        raise ValueError('Adapter seed PLY does not reference the matching reconstructed scene')
    model = source / 'frame000001/sparse/0'
    if not (adapter / 'sparse/0/images.bin').samefile(model / 'images.bin'):
        raise ValueError('Adapter extrinsics do not reference the matching reconstructed scene')
    full_intrinsics = read_intrinsics_binary(model / 'cameras.bin')
    intrinsics = read_intrinsics_binary(adapter / 'sparse/0/cameras.bin')
    extrinsics = read_extrinsics_binary(model / 'images.bin')
    extrinsics_by_name = {extr.name: extr for extr in extrinsics.values()}
    for key, scaled in intrinsics.items():
        original = full_intrinsics[key]
        expected = original.params * [scaled.width / original.width, scaled.height / original.height,
                                      scaled.width / original.width, scaled.height / original.height]
        if scaled.model != 'PINHOLE' or not np.allclose(scaled.params, expected, rtol=0, atol=1e-10):
            raise ValueError('Incorrect separate scaling of fx/fy/cx/cy')
    sizes = set(tuple(size) for size in manifest['half_image_sizes'].values())
    if len(sizes) != 1:
        raise ValueError('Scene has inconsistent camera image sizes')
    size = list(sizes.pop())
    data = DataModule(DataConfig(root=str(adapter), resolution=1, expected_image_size=size,
                                shuffle=False, extra_dataset_kwargs={'num_frames': 300}))
    data.setup()
    expected_names = set(manifest['camera_names'])
    train_names = {item.path.name for item in data.scene.train_dataset.items}
    test_names = {item.path.name for item in data.scene.test_dataset.items}
    if train_names != expected_names - {'cam00.png'} or test_names != {'cam00.png'}:
        raise ValueError('Incorrect held-out/train camera split')
    checked, frames = 0, set()
    for dataset in (data.scene.train_dataset, data.scene.test_dataset):
        if len(dataset) != len({item.path.name for item in dataset.items}) * 300:
            raise ValueError('Incorrect dataset length')
        for index, item in enumerate(dataset.items):
            if item.path.parent.name != f'{item.frame:04d}' or not item.path.is_file():
                raise ValueError('Missing or shifted loader frame: ' + str(item.path))
            if item.frame not in (0, 149, 299):
                continue
            target = source / item.path.stem / 'images' / f'{item.frame:04d}.png'
            if not item.path.samefile(target):
                raise ValueError('Image symlink refers to a different camera/time')
            view = dataset[index]
            with Image.open(target) as image:
                expected = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 255
            if not torch.equal(view.image, expected):
                raise ValueError('Loader resized or changed a canonical training image')
            if list(view.image.shape[-2:]) != size[::-1]:
                raise ValueError('Unexpected loaded image dimensions')
            extr = extrinsics_by_name[item.path.name]
            intr = intrinsics[extr.camera_id]
            expected_fov = [2 * np.arctan(intr.width / (2 * intr.params[0])),
                            2 * np.arctan(intr.height / (2 * intr.params[1]))]
            if not np.allclose([view.fovx, view.fovy], expected_fov, rtol=0, atol=1e-12):
                raise ValueError('Loaded horizontal/vertical field of view differs from fx/fy')
            loaded = view.world_view_transform.numpy().T
            if not np.allclose(loaded[:3, :3], extr.qvec2rotmat(), atol=1e-6):
                raise ValueError('Camera rotation changed in loader')
            if not np.allclose(loaded[:3, 3], extr.tvec, atol=1e-6):
                raise ValueError('Camera translation changed in loader')
            checked += 1
            frames.add(item.frame)
    points = PlyData.read(source / 'points3D_downsample2.ply')['vertex'].data
    xyz = np.column_stack([points[k] for k in ('x', 'y', 'z')])
    if not np.array_equal(data.scene.point_cloud.points, xyz):
        raise ValueError('Loader changed/scaled the 3D initialization coordinates')
    native = load_scene(adapter, resolution=1, split_train_test=True, num_frames=1)
    native.train_dataset.setup()
    native.test_dataset.setup()
    if len(native.train_dataset) != len(expected_names) - 1 or len(native.test_dataset) != 1:
        raise ValueError('Native fresh-frame0 split does not match the dynamic loader')
    return dict(status='passed', source=str(source), adapter=str(adapter), image_size=size,
                frames=300, training_views=len(train_names), test_camera='cam00.png',
                seed_points=len(xyz), sampled_loader_images=checked, sampled_frames=sorted(frames),
                canonical_pngs_equal_loaded_tensors=True, camera_transforms_match_model=True,
                fx_fy_scaled_independently=True, loaded_fov_matches_scaled_fx_fy=True,
                seed_coordinates_unchanged=True,
                native_fresh_frame0_compatible=True, initialization_checkpoint_reused=False,
                metadata_sha256=manifest['metadata_sha256'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--adapter', type=Path, required=True)
    parser.add_argument('--create-adapter', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    result = validate(args.source, args.adapter, args.create_adapter)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
