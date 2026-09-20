"""One-frame renderer-gradient diagnosis; no training pipeline mutations.

Translation, optionally plus axis-angle rotation, is varied; all existing
Gaussian attributes stay fixed. Training cameras only, never cam00. This is
NOT a matched PSNR experiment.
"""
import argparse
from datetime import datetime
import hashlib
import importlib
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch
import yaml
from gradient_ema import GradientEMA

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
P = importlib.import_module('pipeline.ReCon-GS.model.rd_tree_projection')
B = importlib.import_module('pipeline.ReCon-GS.model.rd_tree_basis')
A = importlib.import_module('pipeline.ReCon-GS.model.rd_latent_acceptance')
DATA = importlib.import_module('pipeline.ReCon-GS.data')
MODEL = importlib.import_module('pipeline.ReCon-GS.model.gaussian')
MODULE = importlib.import_module('pipeline.ReCon-GS.module')
CODEC = importlib.import_module('pipeline.ReCon-GS.model.motion_codec')
from libgs.metric import psnr, ssim
from libgs.renderer import Renderer, RendererConfig


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'config/idea_rd_tree_render_probe.yaml')
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    base_path = ROOT / config['base_config']
    base = yaml.safe_load(base_path.read_text())
    if Path(config['experiment_name']).name != config['experiment_name']:
        raise ValueError('experiment_name must be a basename')
    output = ROOT / 'output' / (config['experiment_name'] + '_' + datetime.now().strftime('%Y%m%dT%H%M%S_%f'))
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / 'source_snapshot'
    snapshot.mkdir()
    manifest = []
    for path in (Path(__file__).resolve(), args.config.resolve(), base_path,
                 ROOT / 'scripts/gradient_ema.py',
                 ROOT / 'pipeline/ReCon-GS/model/rd_tree_projection.py',
                 ROOT / 'pipeline/ReCon-GS/model/rd_tree_basis.py',
                 ROOT / 'pipeline/ReCon-GS/model/rd_latent_acceptance.py',
                 ROOT / 'pipeline/ReCon-GS/model/motion_codec.py'):
        shutil.copy2(path, snapshot / path.name)
        manifest.append({'path': str(path.relative_to(ROOT)), 'sha256': digest(path)})
    (output / 'source_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    data_config = dict(base['data'])
    data_config['extra_dataset_kwargs'] = dict(data_config['extra_dataset_kwargs'],
                                               num_frames=int(config['frame']) + 1)
    data = DATA.DataModule(DATA.DataConfig(**data_config))
    # Camera parsing only; no point-cloud conversion or dataset writes.
    data.scene.train_dataset.setup()
    data.current_frame = int(config['frame'])
    dataset = data.get_train_dataset()
    gradient_indices = list(config['gradient_view_indices'])
    check_indices = list(config['check_view_indices'])
    group_size = int(config.get('gradient_group_size', len(gradient_indices)))
    if group_size < 1 or len(gradient_indices) % group_size or len(set(gradient_indices)) != len(gradient_indices):
        raise ValueError('Gradient groups must partition unique view indices evenly')
    order = list(gradient_indices)
    if config.get('shuffle_gradient_groups', False):
        order = np.random.default_rng(config['seed']).permutation(order).tolist()
    gradient_groups = [order[start:start + group_size] for start in range(0, len(order), group_size)]
    if len(gradient_groups) > 1 and (config['curvatures'] or int(config.get('iterations', 0))):
        raise ValueError('Cycling views currently supported only for envelope diagnosis')
    if set(gradient_indices) & set(check_indices):
        raise ValueError('Diagnostic camera groups must be disjoint')
    requested_indices = gradient_indices + check_indices
    # libgs's default 1.6K-resize warning assumes a DataLoader worker exists.
    # Keep the original resolution protocol and use its intended loading path.
    loader = torch.utils.data.DataLoader(torch.utils.data.Subset(dataset, requested_indices),
                                         batch_size=1, num_workers=1,
                                         collate_fn=lambda batch: batch[0])
    views = {index: view.to('cuda') for index, view in zip(requested_indices, loader)}
    for view in views.values():
        if view.path.stem == 'cam00':
            raise ValueError('Test camera must not enter structural diagnosis')
    checkpoint_path = Path(config.get('checkpoint_path', base['ckpt_path']))
    checkpoint = torch.load(checkpoint_path, map_location='cuda', weights_only=False)
    source_frame = int(checkpoint.get('current_frame', -1))
    if source_frame != int(config['frame']) - 1:
        raise ValueError('Probe requires a checkpoint from the immediately preceding frame')
    model = MODEL.GaussianModel(base['module']['sh_degree'])
    model.restore(checkpoint['gaussians'], MODULE.GaussianConfig(**base['module']['gaussian']))
    merged_incremental = 0
    if source_frame > 0:
        if not base['module'].get('merge_to_base', True):
            raise ValueError('Later-frame diagnostic currently requires merge_to_base')
        incremental = MODEL.GaussianModel(base['module']['sh_degree'])
        incremental.restore(checkpoint['gaussians_incr'],
                            MODULE.GaussianConfig(**base['module']['gaussian_stage2']))
        merged_incremental = len(incremental.get_xyz)
        if merged_incremental:
            # Match Module.pre_training_step: previous deformation is already
            # baked at stage1 end; merge previous incremental GS then prune the
            # same number of lowest-opacity GS before starting the next frame.
            model.extend(incremental)
            indices = model.get_opacity.flatten().sort().indices
            model.prune_points(indices[:merged_incremental])
        del incremental
    for name in ('_xyz', '_features_dc', '_features_rest', '_scaling', '_rotation', '_opacity'):
        getattr(model, name).requires_grad_(False)
    renderer = Renderer(RendererConfig(**base['module']['renderer']), model)
    background = torch.zeros(3, device='cuda')
    count = len(model.get_xyz_ori)
    channels = 6 if config.get('rotation', False) else 3
    displacement = torch.zeros((count, channels), dtype=model.get_xyz_ori.dtype,
                               device='cuda', requires_grad=True)
    lambda_ssim = float(base['module']['lambda_dssim'])
    forward_calls, backward_calls = 0, 0

    def forward_loss(field, indices, backward=False):
        nonlocal forward_calls, backward_calls
        tensor = field if torch.is_tensor(field) else torch.as_tensor(field, dtype=torch.float32, device='cuda')
        losses, scores, similarities = [], [], []
        try:
            for index in indices:
                model.delta_xyz = tensor[:, :3]
                model.delta_rot = (CODEC.rotation_vector_to_quaternion(tensor[:, 3:])
                                   if channels == 6 else None)
                view = views[index]
                rendered = renderer(view, background)['render']
                forward_calls += 1
                similarity = ssim(rendered, view.image)
                loss = ((1 - lambda_ssim) * (rendered - view.image).abs().mean()
                        + lambda_ssim * (1 - similarity))
                losses.append(float(loss.detach()))
                scores.append(float(psnr(rendered.detach(), view.image).mean()))
                similarities.append(float(similarity.detach()))
                if backward:
                    (loss / len(indices)).backward()
                    backward_calls += 1
        finally:
            model.delta_xyz = None
            model.delta_rot = None
        return {'distortion': float(np.mean(losses)), 'psnr': float(np.mean(scores)),
                'ssim': float(np.mean(similarities))}

    torch.cuda.synchronize()
    started = time.perf_counter()
    gradient_metrics = forward_loss(displacement, gradient_groups[0], backward=True)
    gradient = displacement.grad.detach().cpu().double().numpy()
    torch.cuda.synchronize()
    gradient_seconds = time.perf_counter() - started
    with torch.no_grad():
        check_metrics = forward_loss(displacement.detach(), check_indices)
    xyz = model.get_xyz_ori.detach().cpu().double().numpy()
    started = time.perf_counter()
    tree = P.build_candidate_tree(xyz, int(config['max_depth']), int(config['min_leaf_size']),
                                  config.get('partition', 'median'),
                                  int(config.get('clustering_iterations', 4)))
    tree_seconds = time.perf_counter() - started
    workspace = P.ProjectionWorkspace(tree) if config.get('optimized_projection', False) else None
    atomic_leaves = sum(node.left < 0 for node in tree.nodes)
    print(f'Loaded {count} Gaussians; {atomic_leaves} candidate leaves; tree {tree_seconds:.3f}s', flush=True)
    gradient_norm = np.linalg.norm(gradient, axis=1)
    input_record = {'scope': __doc__, 'config': config, 'checkpoint': str(checkpoint_path),
                    'source_frame': source_frame, 'merged_incremental': merged_incremental,
                    'checkpoint_sha256': digest(checkpoint_path), 'gaussians': count,
                    'candidate_nodes': len(tree.nodes), 'candidate_leaves': atomic_leaves,
                    'tree_seconds': tree_seconds, 'gradient_seconds': gradient_seconds,
                    'gradient_norm_quantiles': np.quantile(gradient_norm, [0, .5, .9, .99, 1]).tolist(),
                    'gradient_channel_abs_quantiles': np.quantile(np.abs(gradient), [.5, .9, .99, 1], axis=0).tolist(),
                    'gradient_views': [str(views[i].path) for i in gradient_indices],
                    'gradient_groups': gradient_groups,
                    'baseline_gradient_view_indices': gradient_groups[0],
                    'check_views': [str(views[i].path) for i in check_indices],
                    'baseline_gradient_views': gradient_metrics, 'baseline_check_views': check_metrics,
                    'view_sha256': {str(view.path): digest(view.path) for view in views.values()}}
    (output / 'inputs.json').write_text(json.dumps(input_record, indent=2) + '\n')
    np.savez_compressed(output / 'gradient.npz', gradient=gradient.astype(np.float32))
    step = np.broadcast_to(np.asarray(config['quant_step'], dtype=np.float64), (channels,)).copy()
    rows = []
    for rate in config['normalized_rate_weights']:
        actual_rate = float(rate) / count
        baseline = P.project_rd_tree(np.zeros_like(gradient), tree, step, actual_rate)
        for rho in config['curvatures']:
            target = -gradient / float(rho)
            started = time.perf_counter()
            proposal = P.project_rd_tree(target, tree, step, actual_rate, np.full(channels, rho))
            projection_seconds = time.perf_counter() - started
            packet = P.encode_projection(proposal, tree)
            decoded = P.decode_projection(packet, tree)
            assert np.array_equal(decoded, proposal.field)
            started = time.perf_counter()
            with torch.no_grad():
                proposed_gradient = forward_loss(decoded, gradient_indices)
                proposed_check = forward_loss(decoded, check_indices)
            torch.cuda.synchronize()
            evaluation_seconds = time.perf_counter() - started
            delta_bits = proposal.payload_bits - baseline.payload_bits
            actual_change = (proposed_gradient['distortion'] - gradient_metrics['distortion']
                             + actual_rate * delta_bits)
            predicted_change = (float(np.sum(gradient * decoded) + .5 * rho * np.sum(decoded ** 2))
                                + actual_rate * delta_bits)
            packet_name = f'motion_rate{rate}_rho{rho}.rdp'
            (output / packet_name).write_bytes(packet)
            row = {'normalized_lambda': rate, 'rho': rho, 'splits': len(proposal.split_nodes),
                   'payload_bits': proposal.payload_bits, 'packet_bytes': len(packet),
                   'gradient_views': proposed_gradient, 'check_views': proposed_check,
                   'predicted_rd_change': predicted_change, 'actual_rd_change': actual_change,
                   'accepted_by_gradient_views': actual_change < 0,
                   'projection_seconds': projection_seconds, 'evaluation_seconds': evaluation_seconds,
                   'max_displacement': float(np.linalg.norm(decoded, axis=1).max()), 'packet': packet_name}
            rows.append(row)
            with (output / 'proposals.jsonl').open('a') as stream:
                stream.write(json.dumps(row) + '\n')
            print(json.dumps(row), flush=True)
    iterations = []
    if int(config.get('iterations', 0)) > 0:
        rate = float(config['iteration_normalized_rate']) / count
        state = P.project_rd_tree(np.zeros_like(gradient), tree, step, rate)
        for iteration in range(int(config['iterations'])):
            started = time.perf_counter()
            if iteration:
                with torch.no_grad():
                    displacement.copy_(torch.as_tensor(state.field, dtype=displacement.dtype, device='cuda'))
                displacement.grad = None
                current_metrics = forward_loss(displacement, gradient_indices, backward=True)
                current_gradient = displacement.grad.detach().cpu().double().numpy()
            else:
                current_metrics, current_gradient = gradient_metrics, gradient
            evaluated = {state.field.tobytes(): current_metrics}

            def evaluate(field):
                with torch.no_grad():
                    metrics = forward_loss(field, gradient_indices)
                evaluated[field.tobytes()] = metrics
                return metrics['distortion']

            previous_objective = current_metrics['distortion'] + rate * state.payload_bits
            update = P.rd_proposal_step(state, current_gradient, tree,
                                        np.full(channels, config['iteration_base_curvature']), rate,
                                        config['iteration_proposal_scales'], evaluate,
                                        current_metrics['distortion'])
            state = update.projection
            selected_metrics = evaluated[state.field.tobytes()]
            with torch.no_grad():
                selected_check = forward_loss(state.field, check_indices)
            packet = P.encode_projection(state, tree)
            assert np.array_equal(P.decode_projection(packet, tree), state.field)
            packet_name = f'iteration_{iteration:03d}.rdp'
            (output / packet_name).write_bytes(packet)
            leaf_depths = {}

            def collect_depth(index, depth):
                node = tree.nodes[index]
                if index in state.detail_symbols:
                    collect_depth(node.left, depth + 1)
                    collect_depth(node.right, depth + 1)
                else:
                    entry = leaf_depths.setdefault(depth, {'regions': 0, 'gaussians': 0})
                    entry['regions'] += 1
                    entry['gaussians'] += len(node.indices)

            collect_depth(0, 0)
            torch.cuda.synchronize()
            row = {'iteration': iteration, 'accepted': update.accepted,
                   'previous_objective': previous_objective, 'objective': update.objective,
                   'chosen_scale': update.chosen_scale, 'splits': len(state.split_nodes),
                   'payload_bits': state.payload_bits, 'packet_bytes': len(packet),
                   'gradient_views': selected_metrics, 'check_views': selected_check,
                   'leaf_depths': leaf_depths, 'projection_calls': update.projection_calls,
                   'extra_distortion_evaluations': update.distortion_evaluations,
                   'seconds': time.perf_counter() - started, 'packet': packet_name,
                   'trials': update.trials}
            iterations.append(row)
            with (output / 'iterations.jsonl').open('a') as stream:
                stream.write(json.dumps(row) + '\n')
            print('ITERATION ' + json.dumps(row), flush=True)
            if not update.accepted:
                break
    envelope_runs = []
    for settings in config.get('envelope_runs', []):
        gradient_filter = GradientEMA(settings.get('renderer_gradient_ema_beta', 0.0))
        # Explicit detached decoder plus injected renderer gradient is the same
        # first-order estimator as x + stop_gradient(decode(x) - x). The tree
        # envelope gradient is exact a.e.; the renderer STE is not.
        rho = np.broadcast_to(np.asarray(settings['curvature'], dtype=np.float64), (channels,)).copy()
        rate = float(settings['normalized_rate']) / count
        learning_rate = torch.as_tensor(float(settings['step_fraction']) / rho)
        coordinates = settings.get('coordinates', 'point')
        if coordinates not in ('point', 'tree'):
            raise ValueError('Unknown latent coordinates')
        basis = B.TreeMotionBasis(tree, workspace) if coordinates == 'tree' else None
        parameter_count = count if basis is None else basis.coefficient_count
        parameters = torch.zeros((parameter_count, channels), dtype=torch.float64, requires_grad=True)
        optimizer_name = settings.get('optimizer', 'sgd')
        if optimizer_name not in ('sgd', 'adam'):
            raise ValueError('Unknown envelope optimizer')
        optimizer = (torch.optim.Adam([parameters], lr=float(settings['learning_rate']))
                     if optimizer_name == 'adam' else None)
        backtracking = settings.get('rd_backtracking_scales')
        if backtracking is not None and optimizer is not None:
            raise ValueError('Stateless RD acceptance currently supports SGD only')
        decision_views = settings.get('rd_acceptance_views', 'batch')
        if decision_views not in ('batch', 'all_gradient'):
            raise ValueError('Unknown RD acceptance observation set')
        if decision_views != 'batch' and backtracking is None:
            raise ValueError('Fixed observation acceptance requires backtracking')
        decision_cache = A.ObservationLossCache()
        def project_candidate(value):
            candidate_field = value if basis is None else basis(torch.as_tensor(value)).numpy()
            return P.project_rd_tree(candidate_field, tree, step, rate, rho,
                                     vectorized_symbols=config.get('optimized_projection', False),
                                     vectorized_statistics=config.get('optimized_projection', False),
                                     workspace=workspace)
        run_rows = []
        label = str(settings['name'])
        if Path(label).name != label:
            raise ValueError('Envelope run name must be a basename')
        for iteration in range(int(settings['iterations'])):
            started = time.perf_counter()
            latent = parameters if basis is None else basis(parameters)
            envelope, state = P.tree_rate_envelope(latent, tree, step, rate, rho,
                                                   optimized=config.get('optimized_projection', False),
                                                   workspace=workspace)
            projection_seconds = time.perf_counter() - started
            backward_started = time.perf_counter()
            current_group = gradient_groups[iteration % len(gradient_groups)]
            if iteration == 0:
                current_metrics, current_gradient = gradient_metrics, gradient
            else:
                with torch.no_grad():
                    displacement.copy_(torch.as_tensor(state.field, dtype=displacement.dtype, device='cuda'))
                displacement.grad = None
                current_metrics = forward_loss(displacement, current_group, backward=True)
                current_gradient = displacement.grad.detach().cpu().double().numpy()
            backward_seconds = time.perf_counter() - backward_started
            with torch.no_grad():
                current_check = forward_loss(state.field, check_indices)
            packet = P.encode_projection(state, tree)
            assert np.array_equal(P.decode_projection(packet, tree), state.field)
            packet_name = f'envelope_{label}_{iteration:03d}.rdp'
            (output / packet_name).write_bytes(packet)
            optimization_started = time.perf_counter()
            envelope.backward(retain_graph=basis is not None)
            filtered_gradient = gradient_filter.update(current_gradient)
            latent.backward(torch.as_tensor(filtered_gradient, dtype=latent.dtype))
            acceptance = None
            if optimizer is None:
                with torch.no_grad():
                    normalization = 1. if basis is None else basis.energy_weights[:, None]
                    direction = -learning_rate * parameters.grad / normalization
                    if backtracking is None:
                        parameters += direction
                    else:
                        decision_indices = gradient_indices if decision_views == 'all_gradient' else current_group
                        observation_key = (int(config['frame']), *decision_indices)
                        baseline_evaluations = 0
                        base_distortion = current_metrics['distortion']
                        if decision_views == 'all_gradient':
                            base_distortion = decision_cache.get(packet, observation_key)
                            if base_distortion is None:
                                base_distortion = forward_loss(state.field, decision_indices)['distortion']
                                baseline_evaluations = 1
                        decision = A.accept_latent_step(
                            parameters.detach().numpy(), direction.numpy(), state, tree, rate,
                            backtracking, project_candidate,
                            lambda field: forward_loss(field, decision_indices)['distortion'],
                            base_distortion)
                        parameters.copy_(torch.as_tensor(decision.latent))
                        if decision_views == 'all_gradient':
                            decision_cache.put(P.encode_projection(decision.projection, tree),
                                               observation_key, decision.distortion)
                        acceptance = dict(accepted=decision.accepted, scale=decision.scale,
                                          decision_view_indices=decision_indices,
                                          baseline_distortion_evaluations=baseline_evaluations,
                                          extra_single_view_forwards=(baseline_evaluations + decision.distortion_evaluations)
                                          * len(decision_indices),
                                          baseline_objective=base_distortion + rate * state.payload_bits,
                                          accepted_objective=decision.objective,
                                          extra_distortion_evaluations=decision.distortion_evaluations,
                                          extra_projection_calls=decision.projection_calls,
                                          trials=decision.trials)
            else:
                # Adam preconditions optimization of x, not the coding metric.
                # The projection still uses its shared-per-channel rho, so its
                # orthogonal tree decomposition is unchanged.
                optimizer.step()
            parameters.grad = None
            optimization_seconds = time.perf_counter() - optimization_started
            row = {'name': label, 'iteration': iteration, 'settings': settings,
                   'updates_before_evaluation': iteration,
                   'latent_coordinates': coordinates, 'latent_scalar_parameters': parameters.numel(),
                   'rd_acceptance': acceptance,
                   'gradient_view_indices': current_group,
                   'renderer_gradient_ema_beta': gradient_filter.beta,
                   'renderer_gradient_norm': float(np.linalg.norm(current_gradient)),
                   'filtered_renderer_gradient_norm': float(np.linalg.norm(filtered_gradient)),
                   'gradient_metrics_comparable_across_steps': len(gradient_groups) == 1,
                   'gradient_views': current_metrics, 'check_views': current_check,
                   'objective': current_metrics['distortion'] + rate * state.payload_bits,
                   'envelope': float(envelope.detach()),
                   'commitment': float(envelope.detach()) - rate * state.payload_bits,
                   'payload_bits': state.payload_bits, 'packet_bytes': len(packet),
                   'splits': len(state.split_nodes), 'packet': packet_name,
                   'projection_seconds': projection_seconds,
                   'training_backward_seconds': backward_seconds,
                   'optimizer_seconds': optimization_seconds,
                   'max_translation': float(np.linalg.norm(state.field[:, :3], axis=1).max()),
                   'max_rotation_radians': (float(np.linalg.norm(state.field[:, 3:], axis=1).max())
                                            if channels == 6 else 0.),
                   'seconds': time.perf_counter() - started}
            run_rows.append(row)
            with (output / 'envelopes.jsonl').open('a') as stream:
                stream.write(json.dumps(row) + '\n')
            print('ENVELOPE ' + json.dumps(row), flush=True)
        envelope_runs.append({'settings': settings, 'states': run_rows})
    result = dict(input_record, proposals=rows, iterations=iterations, envelope_runs=envelope_runs,
                  backward_evaluations=backward_calls, render_evaluations=forward_calls,
                  extra_render_evaluations=forward_calls - backward_calls)
    (output / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    print('Completed diagnostic:', output, flush=True)


if __name__ == '__main__':
    main()
