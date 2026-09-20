"""Frozen-latent structural counterfactuals on training cameras only.

Replay from an earlier checkpoint to recover the exact pre-motion scene. Do not
apply counterfactuals to the already-deformed target checkpoint. No optimization,
no current-frame incremental GS, no test-camera selection; not a full-run score.
"""
import argparse
from datetime import datetime
import hashlib
import importlib
import json
from pathlib import Path
import sys
import time
import zipfile

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
P = importlib.import_module('pipeline.ReCon-GS.model.rd_tree_projection')
M = importlib.import_module('pipeline.ReCon-GS.model.rd_tree_motion')
G = importlib.import_module('pipeline.ReCon-GS.model.gaussian')
C = importlib.import_module('pipeline.ReCon-GS.module')
D = importlib.import_module('pipeline.ReCon-GS.data')
CODEC = importlib.import_module('pipeline.ReCon-GS.model.motion_codec')
from libgs.metric import psnr, ssim
from libgs.renderer import Renderer, RendererConfig


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--frames', type=int, nargs='+', default=[10, 50, 99])
    args = parser.parse_args()
    run = args.run.resolve()
    config = yaml.safe_load((run / 'config.yaml').read_text())
    if not config['module']['merge_to_base']:
        raise ValueError('Requires merge-to-base protocol')
    output = ROOT / 'output' / ('idea_frozen_tree_cuts_' + datetime.now().strftime('%Y%m%dT%H%M%S_%f'))
    output.mkdir(exist_ok=False)
    sources = [Path(__file__).resolve(), *sorted((ROOT / 'pipeline').rglob('*.py'))]
    source_hashes = {}
    with zipfile.ZipFile(output / 'sources.zip', 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sources:
            archive.write(path, str(path.relative_to(ROOT)))
            source_hashes[str(path.relative_to(ROOT))] = digest(path)
        archive.write(run / 'config.yaml', 'run_config.yaml')
    hashes = {str(run / 'config.yaml'): digest(run / 'config.yaml')}
    data = D.DataModule(D.DataConfig(**config['data']))
    data.scene.train_dataset.setup()
    storage = run / 'frame_storage'
    steps = config['trainer']['num_incr_steps']
    checkpoints = {0: Path(config['ckpt_path'])}
    for path in run.glob(f'ckpt-*-{steps}.pth'):
        checkpoints[int(path.name.split('-')[1])] = path
    records = []
    started = time.perf_counter()
    for frame in args.frames:
        if frame not in checkpoints or frame < 1:
            raise ValueError(f'Missing target checkpoint {frame}')
        previous = max(f for f in checkpoints if f < frame)
        start_path = checkpoints[previous]
        hashes[str(start_path)] = digest(start_path)
        saved = torch.load(start_path, map_location='cuda', weights_only=False)
        if saved['current_frame'] != previous:
            raise ValueError('Starting checkpoint frame mismatch')
        model = G.GaussianModel(config['module']['sh_degree'])
        model.restore(saved['gaussians'], C.GaussianConfig(**config['module']['gaussian']))
        incremental = None
        if previous:
            incremental = G.GaussianModel(config['module']['sh_degree'])
            incremental.restore(saved['gaussians_incr'], C.GaussianConfig(**config['module']['gaussian_stage2']))
        del saved
        for current in range(previous + 1, frame + 1):
            if incremental is not None and len(incremental.get_xyz):
                count = len(incremental.get_xyz)
                model.extend(incremental)
                model.prune_points(model.get_opacity.flatten().sort().indices[:count])
            if current == frame:
                break
            packet_path = storage / f'motion_{current:03d}.rdp'
            protocol_path = packet_path.with_suffix('.rdp.json')
            protocol = json.loads(protocol_path.read_text())
            decoded = M.decode_tree_motion(packet_path.read_bytes(),
                                           model.get_xyz_ori.detach().cpu().double().numpy(), protocol)
            value = torch.as_tensor(decoded, device='cuda', dtype=model.get_xyz_ori.dtype)
            model.apply_deformation(value[:, :3], CODEC.rotation_vector_to_quaternion(value[:, 3:]))
            incr_path = storage / f'incremental_{current:03d}.ply'
            incremental = G.GaussianModel(config['module']['sh_degree'])
            incremental.load_ply(incr_path)
            for path in (packet_path, protocol_path, incr_path):
                hashes[str(path)] = digest(path)
        target_path = checkpoints[frame]
        hashes[str(target_path)] = digest(target_path)
        saved = torch.load(target_path, map_location='cpu', weights_only=False)
        if saved['current_frame'] != frame:
            raise ValueError('Target checkpoint frame mismatch')
        state = saved['deformation']
        del saved
        xyz = model.get_xyz_ori.detach().cpu().double()
        torch.testing.assert_close(xyz, state['state']['xyz'].double(), rtol=0, atol=0)
        settings = M.RDTreeConfig(**state['config'])
        tree = P.build_candidate_tree(xyz.numpy(), settings.max_depth, settings.min_leaf_size,
                                     settings.partition, settings.clustering_iterations)
        workspace = P.ProjectionWorkspace(tree)
        latent = state['state']['latent'].double().numpy()
        rho = state['state']['rho'].double().numpy()
        step = [settings.quant_xyz] * 3 + [settings.quant_rotation] * 3
        rate = settings.normalized_rate / len(xyz)
        packet_path = storage / f'motion_{frame:03d}.rdp'
        hashes[str(packet_path)] = digest(packet_path)
        packet = packet_path.read_bytes()
        original = P.project_rd_tree(latent, tree, step, rate, rho, vectorized_symbols=True,
                                     vectorized_statistics=True, workspace=workspace,
                                     fixed_cut_depth=settings.fixed_cut_depth)
        if P.encode_projection(original, tree) != packet or packet != state['packet']:
            raise ValueError('Frozen state does not reproduce the saved motion packet')
        for name in ('_xyz', '_features_dc', '_features_rest', '_scaling', '_rotation', '_opacity'):
            getattr(model, name).requires_grad_(False)
        data.current_frame = frame
        dataset = data.get_train_dataset()
        loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=1,
                                             collate_fn=lambda batch: batch[0])
        views = [view.to('cuda') for view in loader]
        if len(views) != 20 or len({str(v.path) for v in views}) != 20 or any(v.path.stem == 'cam00' for v in views):
            raise ValueError('Expected exactly 20 unique training cameras, no cam00')
        for view in views:
            hashes[str(view.path)] = digest(view.path)
        renderer = Renderer(RendererConfig(**config['module']['renderer']), model)
        background = torch.zeros(3, device='cuda')
        for cut in (None, 13, 16):
            projection = P.project_rd_tree(latent, tree, step, rate, rho, vectorized_symbols=True,
                                          vectorized_statistics=True, workspace=workspace,
                                          fixed_cut_depth=cut)
            encoded = P.encode_projection(projection, tree)
            decoded = P.decode_projection(encoded, tree)
            if not np.array_equal(decoded, projection.field):
                raise AssertionError('Counterfactual packet round trip')
            metrics = []
            with torch.no_grad():
                tensor = torch.as_tensor(decoded, dtype=model.get_xyz_ori.dtype, device='cuda')
                model.delta_xyz = tensor[:, :3]
                model.delta_rot = CODEC.rotation_vector_to_quaternion(tensor[:, 3:])
                try:
                    for view in views:
                        rendered = renderer(view, background)['render']
                        similarity = ssim(rendered, view.image)
                        distortion = ((1 - config['module']['lambda_dssim']) * (rendered - view.image).abs().mean()
                                      + config['module']['lambda_dssim'] * (1 - similarity))
                        metrics.append(dict(path=str(view.path), distortion=float(distortion),
                                            psnr=float(psnr(rendered, view.image).mean()), ssim=float(similarity)))
                finally:
                    model.delta_xyz = model.delta_rot = None
            means = {key: float(np.mean([m[key] for m in metrics])) for key in ('distortion', 'psnr', 'ssim')}
            row = dict(frame=frame, replay_start=previous, fixed_cut_depth=cut, splits=len(projection.split_nodes),
                       payload_bits=projection.payload_bits, packet_bytes=len(encoded),
                       proxy_objective=projection.objective, quadratic_error=projection.distortion,
                       image_rd=means['distortion'] + rate * projection.payload_bits,
                       means=means, per_view=metrics, pre_motion_xyz_exact=True, source_packet_exact=True,
                       single_view_forwards=len(views), backward_calls=0)
            if not all(np.isfinite(v) for v in [row['image_rd'], row['proxy_objective'], *means.values()]):
                raise ValueError('Nonfinite counterfactual metrics')
            records.append(row)
            with (output / 'records.jsonl').open('a') as stream:
                stream.write(json.dumps(row) + '\n')
            print(json.dumps({k: v for k, v in row.items() if k != 'per_view'}), flush=True)
        del renderer, views, model, incremental, state, tree, workspace
    (output / 'report.json').write_text(json.dumps(dict(run=str(run), scope=__doc__, frames=args.frames,
        records=records, hashes=hashes, source_hashes=source_hashes, wall_seconds=time.perf_counter()-started), indent=2))
    print('Saved:', output, flush=True)


if __name__ == '__main__':
    main()
