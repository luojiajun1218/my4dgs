"""CPU-only layout audit on an existing, untouched shared first-frame reference.

Compare the default layout with optional robust statistics support. Also apply
the local S2GS grid/LoD rules to the SAME reference points for diagnosis, not
as a reproduction of S2GS's COLMAP initialization or a rendering comparison.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
LAYOUT = importlib.import_module('pipeline.model.spatial_gate')
MOTION = importlib.import_module('pipeline.model.spatial_gated_motion')
RIGID = importlib.import_module('pipeline.model.rigid_residual_tree')
CODEC = importlib.import_module('pipeline.model.rigid_tree_codec')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def quantiles(values):
    return {str(q): float(np.quantile(values, q)) for q in (0., .5, .95, .99, .999, 1.)}


def layout_stats(xyz, paths, parents, min_region_size):
    paths, parents = paths.numpy(), parents.numpy()
    count = len(parents)
    population = np.bincount(paths.ravel(), minlength=count)
    child_counts = np.bincount(parents[np.flatnonzero(population)], minlength=count)
    leaf = np.bincount(paths[:, -1], minlength=count)
    occupied = np.flatnonzero(leaf)
    lower = np.full((count, 3), np.inf)
    upper = np.full((count, 3), -np.inf)
    np.minimum.at(lower, paths[:, -1], xyz)
    np.maximum.at(upper, paths[:, -1], xyz)
    diagonal = np.linalg.norm(upper[occupied] - lower[occupied], axis=1)
    return dict(candidate_nodes=count, point_count=len(xyz),
                occupied_leaf_count=len(occupied),
                occupied_leaf_population=quantiles(leaf[occupied]),
                occupied_leaf_point_aabb_diagonal=quantiles(diagonal),
                eligible_gates=int(((population >= min_region_size) & (child_counts >= 2)).sum()),
                occupied_nodes_per_level=[len(np.unique(paths[:, level])) for level in range(paths.shape[1])])


def s2gs_grid_stats(xyz, camera_centers, config, support):
    """Independent NumPy implementation of local S2GS init rules, geometry only."""
    ratio = config['dist_ratio']
    distances = np.linalg.norm(xyz[:, None, :].astype(np.float64) - camera_centers[None], axis=-1)
    per_camera = np.quantile(distances, [1 - ratio, ratio], axis=0).T.reshape(-1)
    dist_min, standard_dist = np.quantile(per_camera, [1 - ratio, ratio])
    levels = config['levels']
    if levels == -1:
        levels = int(np.rint(np.log(standard_dist / dist_min) / np.log(config['log_base']))) + 1
    box_min = float(xyz.min()) * config['extend']
    box_max = float(xyz.max()) * config['extend']
    base_layer = config['base_layer']
    if base_layer < 0:
        base_layer = int(np.rint(np.log2((box_max - box_min) / .02))) - levels // 2 + 1
    coarse_size = (box_max - box_min) / config['fork'] ** base_layer
    layer_stats = []
    for level in range(levels):
        size = coarse_size / config['fork'] ** level
        cells = np.floor((xyz.astype(np.float64) - box_min) / size).astype(np.int64)
        unique = np.unique(cells, axis=0)
        core_count = len(np.unique(cells[support], axis=0))
        positions = unique * size + box_min + config['padding'] * size
        dist = np.linalg.norm(positions[:, None, :] - camera_centers[None], axis=-1)
        predicted = np.clip(np.rint(np.log(standard_dist / dist) / np.log(config['log_base'])), 0, levels - 1)
        retained = (predicted >= level).mean(1) > config['visible_threshold']
        layer_stats.append(dict(level=level, voxel_size=float(size),
                                occupied_voxels=len(unique), core_occupied_voxels=core_count,
                                retained_by_distance_lod=int(retained.sum())))
    return dict(scope='Local S2GS grid/LoD rules on our shared GS, not its original COLMAP input',
                camera_scope='Unique training camera centers, cam00 excluded, image scale1',
                global_box=[box_min, box_max], coarse_voxel_size=coarse_size,
                distance_min=float(dist_min), standard_distance=float(standard_dist),
                levels=levels, layers=layer_stats)


def codec_check(xyz, rotation, extent, depth, output):
    config = RIGID.RigidTreeConfig()
    config.spatial_gate.max_depth = depth
    config.spatial_gate.initialization = 'robust_median'
    model = MOTION.SpatialGatedMotion(config)
    model.setup(xyz, rotation, extent, frame_index=1)
    reference = copy.deepcopy(model.frame_reference_state)
    with torch.no_grad():
        model.gs_translation[::100, 0] = model.gs_steps[0] * 7
        model.gs_rotation[::200, 1] = model.gs_steps[3] * 3
    fitted_xyz, fitted_q = model.materialized_pose()
    model.register_materialized_layout(fitted_xyz, fitted_q)
    info = model.export_motion(output / 'robust_layout_synthetic_motion.bin')
    decoded = CODEC.decode_motion(Path(info['path']).read_bytes(), reference)
    torch.testing.assert_close(decoded.xyz, fitted_xyz, rtol=0, atol=1e-6)
    torch.testing.assert_close(decoded.rotation, fitted_q, rtol=0, atol=1e-6)
    resumed = MOTION.SpatialGatedMotion(config)
    resumed.restore(copy.deepcopy(model.capture()))
    for current in (model, resumed):
        current.setup(decoded.xyz, decoded.rotation, extent, frame_index=2)
    torch.testing.assert_close(model.spatial_paths, resumed.spatial_paths, rtol=0, atol=0)
    return dict(scope='Synthetic sparse motion on all real reference GS, not a trained frame',
                decoded_pose_matches=True, resume_matches=True,
                extra_per_frame_codec_fields=False,
                source_xyz_unchanged=bool(torch.equal(reference.xyz, xyz)),
                motion_bytes=Path(info['path']).stat().st_size)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--s2gs-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(2)
    reference = args.run / 'frame_storage/rigid_tree/reference_frame0.npz'
    before = sha256(reference)
    config = yaml.safe_load((args.run / 'config.yaml').read_text())['module']['rigid_tree']['spatial_gate']
    depth, min_region_size = config['max_depth'], config['min_region_size']
    with np.load(reference, allow_pickle=False) as packet:
        xyz = torch.from_numpy(packet['xyz'].copy())
        rotation = torch.from_numpy(packet['rotation'].copy())
        extent = float(packet['extent'])
        saved_paths = packet['spatial_paths'].copy()
        saved_centers = packet['spatial_centers'].copy()
    support = LAYOUT.spatial_initialization_support(xyz, 'robust_median').numpy()
    positions = xyz.numpy()
    center = np.median(positions.astype(np.float64), axis=0)
    radii = np.linalg.norm(positions.astype(np.float64) - center, axis=1)
    pushed = xyz.clone()
    pushed[~torch.from_numpy(support)] = torch.as_tensor(center, dtype=xyz.dtype) + 1000 * (
        pushed[~torch.from_numpy(support)] - torch.as_tensor(center, dtype=xyz.dtype))
    report = dict(scope='CPU geometry and codec only; no training, image quality or rendering FPS',
                  reference=str(reference.resolve()), reference_sha256=before,
                  point_count=len(xyz), radial_distances=quantiles(radii),
                  robust_support_count=int(support.sum()), excluded_from_statistics=int((~support).sum()),
                  excluded_from_model=0, robustness_stress='Move unsupported points1000times farther, keeping all GS',
                  layouts={})
    for mode in ('median', 'robust_median'):
        paths, parents, opened, centers = LAYOUT.spatial_layout(xyz, depth, True, initialization=mode)
        if mode == 'median':
            assert np.array_equal(paths.numpy(), saved_paths)
            assert np.array_equal(centers.numpy(), saved_centers)
            report['default_matches_original_reference_exactly'] = True
        routed = LAYOUT.spatial_route(xyz, centers, depth)
        torch.testing.assert_close(routed, paths, rtol=0, atol=0)
        after = LAYOUT.spatial_layout(pushed, depth, True, initialization=mode)
        report['layouts'][mode] = dict(
            **layout_stats(positions, paths, parents, min_region_size),
            statistics_bounds=np.stack((positions[support].min(0), positions[support].max(0))).tolist()
                if mode == 'robust_median' else np.stack((positions.min(0), positions.max(0))).tolist(),
            source_routing_exact=True,
            stress_changed_split_planes=int((centers != after[3]).any(1).sum()),
            stress_changed_supported_paths=int(((paths != after[0]).any(1).numpy() & support).sum()))
        print(f'{mode}: {report["layouts"][mode]}', flush=True)

    cameras = json.loads((args.run / 'cameras.json').read_text())
    unique = {item['img_name']: item['position'] for item in cameras if item['img_name'] != 'cam00'}
    camera_centers = np.asarray([unique[name] for name in sorted(unique)], dtype=np.float64)
    assert len(camera_centers) == 20, 'Audit expects the existing20training-camera split'
    s2gs_config_path = args.s2gs_root / 'configs/dynerf_flame_steak.yaml'
    s2gs_config = yaml.safe_load(s2gs_config_path.read_text())['opt_params_lod']
    report['s2gs_source'] = dict(
        implementation=str(args.s2gs_root / 'scene/gaussian_model.py'),
        implementation_sha256=sha256(args.s2gs_root / 'scene/gaussian_model.py'),
        config=str(s2gs_config_path), config_sha256=sha256(s2gs_config_path))
    report['s2gs_literal_grid'] = s2gs_grid_stats(positions, camera_centers, s2gs_config, support)
    report['s2gs_literal_grid_stress'] = s2gs_grid_stats(pushed.numpy(), camera_centers, s2gs_config, support)
    args.output.mkdir(parents=True)
    report['real_point_codec_check'] = codec_check(xyz, rotation, extent, depth, args.output)
    assert before == sha256(reference), 'Historical reference was modified'
    report['historical_reference_unchanged'] = True
    (args.output / 'audit.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(dict(output=str(args.output.resolve()),
                          support_count=int(support.sum()), geometry_only=True,
                          source_unchanged=True)), flush=True)


if __name__ == '__main__':
    main()
