"""Fixed-state, image-blind stratified KEEP candidate quality/gradient audit."""
import argparse
import csv
import hashlib
import importlib
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from libgs.metric import psnr, ssim
from libgs.renderer import Renderer
from libgs.utils.config import to_structured


def sample_keep_nodes(selection, layout, seed, count, bins):
    """Sample by membership size before any image is read or rendered."""
    rng = np.random.default_rng(seed)
    result, populations = [], []
    for level, states in enumerate(selection['states']):
        sizes = torch.bincount(layout['index'][level], minlength=len(states)).cpu().numpy()
        eligible = torch.where(selection['visited'][level] & (states == 0))[0].cpu().numpy()
        eligible = eligible[sizes[eligible] > 0]
        order = eligible[np.lexsort((eligible, sizes[eligible]))]
        groups = np.array_split(order, bins)
        populations.append(dict(level=level, reachable_keep_nodes=len(order)))
        for size_bin, group in enumerate(groups):
            requested = count // bins + int(size_bin < count % bins)
            chosen = rng.choice(group, min(requested, len(group)), replace=False)
            for node in sorted(map(int, chosen)):
                result.append(dict(level=level, node=node, size_bin=size_bin,
                                   members=int(sizes[node])))
    return result, populations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    if 'cam00.png' in cfg['camera_names']:
        raise ValueError('Passive cam00 must not select or diagnose the intervention')
    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    torch.set_num_threads(2)
    package = importlib.import_module('pipeline.ReCon-GS')
    module = importlib.import_module('pipeline.ReCon-GS.model.gated_residual')
    gaussian = importlib.import_module('pipeline.ReCon-GS.model.gaussian')
    structured, unused = to_structured(OmegaConf.load(cfg['source_config']), package.Config)
    if unused:
        raise ValueError(f'Unknown source config: {unused}')
    checkpoint = Path(cfg['checkpoint'])
    packet = torch.load(checkpoint, map_location='cuda', weights_only=False)
    frame = int(packet['current_frame'])
    saved = packet['quantized_residual_motion']
    motion = module.GatedResidualMotion(package.ModuleConfig().quantized_residual.__class__(**saved['config']))
    motion.restore(saved)
    selection = motion.gate_state_arrays()
    states = [value.clone() for value in selection['states']]
    samples, populations = sample_keep_nodes(selection, motion.active_layout, cfg['seed'],
                                             cfg['samples_per_level'], cfg['size_bins'])
    run = Path(cfg['output_root']) / cfg['experiment_name'] / datetime.now().strftime('%Y%m%dT%H%M%S')
    run.mkdir(parents=True, exist_ok=False)
    (run / 'config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    (run / 'sample_plan.json').write_text(json.dumps(dict(checkpoint_sha256=hashlib.sha256(
        checkpoint.read_bytes()).hexdigest(), frame=frame, populations=populations, samples=samples,
        rule='Fixed seed; split reachable KEEP by membership-size rank into3 bins; sample8/level.',
        frozen_before_any_images=True), indent=2))
    print(f'Audit run: {run}; frame{frame}; samples{len(samples)}', flush=True)
    motion.committed = False
    motion.frozen_states = None
    for parameter in motion.parameters():
        parameter.requires_grad_(True)
    if any(not torch.equal(a, b) for a, b in zip(states, motion.gate_state_arrays()['states'])):
        raise ValueError('Unfreezing checkpoint changed hard decisions')
    model = gaussian.GaussianModel(structured.module.sh_degree)
    model.restore(packet['gaussians'], structured.module.gaussian)
    for parameter in (model._xyz, model._rotation, model._opacity, model._scaling,
                      model._features_dc, model._features_rest):
        parameter.requires_grad_(False)
    model.noise_scale = None
    renderer = Renderer(structured.module.renderer, model)
    background = torch.ones(3, device='cuda') if structured.data.white_background else torch.zeros(3, device='cuda')
    old_count = len(motion.base_xyz)
    final_xyz = model.get_xyz_ori.detach()
    final_rotation = model.get_rotation_ori.detach()
    with torch.no_grad():
        xyz, rotation = motion(cfg['gradient_step'])
    torch.testing.assert_close(xyz, final_xyz[:old_count], rtol=1e-5, atol=1e-6)
    # Motion stores raw Gaussian quaternions; the renderer normalizes them.
    torch.testing.assert_close(torch.nn.functional.normalize(rotation, dim=-1),
                               final_rotation[:old_count], rtol=1e-5, atol=1e-6)
    structured.data.extra_dataloader_kwargs.update(num_workers=cfg['workers'])
    data = package.DataModule(structured.data)
    data.setup()
    data.current_frame = frame
    views = [now for now, _ in data.selective_training_pairs(cfg['camera_names'], 'cuda', 0.)]
    validation = set(structured.data.validation_camera_names)
    training = set(structured.data.training_camera_names)
    if not set(cfg['camera_names']).issubset(training | validation):
        raise ValueError('Views outside configured train/validation split')
    parameters = [*motion.region_translation, *motion.region_rotation, *motion.gate_logits]
    weight = structured.module.lambda_dssim
    renders = 0
    seconds = dict(proxy=0., candidates=0.)

    def image_loss(view, xyz, rotation):
        nonlocal renders
        model.rigid_xyz = torch.cat((xyz, final_xyz[old_count:]), 0)
        model.rigid_rotation = torch.cat((rotation, final_rotation[old_count:]), 0)
        image = renderer(view, background)['render']
        renders += 1
        loss = (1 - weight) * torch.nn.functional.l1_loss(image, view.image) + weight * (1 - ssim(image, view.image))
        return loss, image

    rows, baseline = [], []
    for view in views:
        torch.cuda.synchronize()
        started = time.perf_counter()
        xyz, rotation = motion(cfg['gradient_step'], training=True)
        loss, image = image_loss(view, xyz, rotation)
        gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
        proxy_loss = float(loss.detach())
        base_psnr = float(psnr(image.detach().clamp(0., 1.), view.image))
        baseline.append(dict(camera=view.path.name, split='validation' if view.path.name in validation else 'train',
                             photometric_loss=proxy_loss, psnr=base_psnr))
        del loss, image, xyz, rotation
        torch.cuda.synchronize()
        seconds['proxy'] += time.perf_counter() - started
        for item in samples:
            level, node = item['level'], item['node']
            true_states = [value.clone() for value in states]
            true_states[level][node] = module.SHARE
            dense = motion._dense_symbols()
            steps = motion.motion_steps_vector
            regions = [dense[f'region_{index}'] * steps for index in range(3)]
            local = dense['gs_local'] * steps
            torch.cuda.synchronize()
            started = time.perf_counter()
            xyz, rotation = module._motion(motion.base_xyz, motion.base_rotation, motion.active_layout,
                                            regions, local, true_states)
            candidate_loss, candidate_image = image_loss(view, xyz, rotation)
            true_gradient = torch.autograd.grad(candidate_loss, [motion.region_translation[level],
                motion.region_rotation[level]], allow_unused=False)
            true = torch.cat((true_gradient[0][node], true_gradient[1][node])).detach()
            proxy = torch.cat((gradients[level][node], gradients[3 + level][node])).detach()
            gate = gradients[6 + level][node].detach()
            true_norm, proxy_norm = float(true.norm()), float(proxy.norm())
            dot = float(torch.dot(true, proxy))
            cosine = dot / (true_norm * proxy_norm) if min(true_norm, proxy_norm) > cfg['gradient_norm_epsilon'] else None
            delta = float(candidate_loss.detach()) - proxy_loss
            gate_direction = float(gate[module.SHARE] - gate[module.KEEP])
            meaningful = abs(delta) > cfg['loss_change_epsilon'] and abs(gate_direction) > cfg['gate_direction_epsilon']
            rows.append(dict(**item, camera=view.path.name,
                split='validation' if view.path.name in validation else 'train',
                keep_loss=proxy_loss, candidate_loss=float(candidate_loss.detach()), loss_change=delta,
                keep_psnr=base_psnr, candidate_psnr=float(psnr(candidate_image.detach().clamp(0., 1.), view.image)),
                gate_share_minus_keep_gradient=gate_direction, gate_comparison_meaningful=meaningful,
                gate_direction_agrees=(delta * gate_direction > 0) if meaningful else None,
                worse_candidate_rewarded=(delta > cfg['loss_change_epsilon'] and gate_direction < -cfg['gate_direction_epsilon']),
                better_candidate_discouraged=(delta < -cfg['loss_change_epsilon'] and gate_direction > cfg['gate_direction_epsilon']),
                true_motion_gradient=true.cpu().tolist(), proxy_motion_gradient=proxy.cpu().tolist(),
                true_motion_gradient_norm=true_norm, proxy_motion_gradient_norm=proxy_norm,
                motion_gradient_dot=dot, motion_gradient_cosine=cosine,
                proxy_step_increases_candidate_loss=(dot < 0) if cosine is not None else None))
            del candidate_loss, candidate_image, xyz, rotation, dense, regions, local, true_gradient
            torch.cuda.synchronize()
            seconds['candidates'] += time.perf_counter() - started
        print(f"{view.path.name}: completed{len(samples)} candidate renders", flush=True)
        # Keep completed camera evidence if later camera fails.
        (run / 'rows.json').write_text(json.dumps(rows, indent=2))
    aggregates = {}
    for split in ('train', 'validation'):
        selected = [row for row in rows if row['split'] == split]
        gate_selected = [row for row in selected if row['gate_comparison_meaningful']]
        motion_selected = [row for row in selected if row['motion_gradient_cosine'] is not None]
        aggregates[split] = dict(samples=len(selected), meaningful_gate_pairs=len(gate_selected),
            gate_direction_agreement=statistics.fmean(row['gate_direction_agrees'] for row in gate_selected) if gate_selected else None,
            worse_candidates_rewarded=sum(row['worse_candidate_rewarded'] for row in selected),
            better_candidates_discouraged=sum(row['better_candidate_discouraged'] for row in selected),
            positive_candidate_loss_changes=sum(row['loss_change'] > cfg['loss_change_epsilon'] for row in selected),
            negative_candidate_loss_changes=sum(row['loss_change'] < -cfg['loss_change_epsilon'] for row in selected),
            mean_candidate_loss_change=statistics.fmean(row['loss_change'] for row in selected),
            meaningful_motion_pairs=len(motion_selected),
            opposing_motion_gradients=sum(row['proxy_step_increases_candidate_loss'] for row in motion_selected),
            mean_motion_cosine=statistics.fmean(row['motion_gradient_cosine'] for row in motion_selected) if motion_selected else None)
    summary = dict(run=str(run), checkpoint=str(checkpoint), frame=frame, sample_plan='sample_plan.json',
        unchanged_checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        baseline=baseline, aggregates=aggregates, render_count=renders, backward_count=renders,
        seconds=seconds, parameters_updated=False,
        limitations=['Checkpoint is completed frame5, not every optimization step or later frames.',
            'Membership-stratified sample; no outcome-based region selection or masks.',
            'One-node discrete SHARE loss difference versus local logit gradient is an approximation comparison.',
            'Candidate-own versus proxy motion dot directly tests first-order candidate update direction.',
            'Photometric gradients exclude gate storage penalties; all birth attributes are fixed.',
            'Frozen gates are temporarily made differentiable in memory; hard states/tensors preserved.',
            'cam00 excluded; validation is diagnostic, not used to fit or select samples.'])
    (run / 'summary.json').write_text(json.dumps(summary, indent=2))
    scalar_rows = [{key: value for key, value in row.items() if not isinstance(value, list)} for row in rows]
    with (run / 'samples.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(scalar_rows[0]))
        writer.writeheader()
        writer.writerows(scalar_rows)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
