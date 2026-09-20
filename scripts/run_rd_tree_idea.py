"""Run bounded analytic experiments and archive their exact inputs/sources."""
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
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
P = importlib.import_module('pipeline.ReCon-GS.model.rd_tree_projection')


def coupled_optimization(config, seed, curvature_multiplier, mode):
    rng = np.random.default_rng(seed)
    count = int(config['proximal_points'])
    matrix = rng.normal(size=(2 * count, count)) / np.sqrt(2 * count)
    truth = np.ones(count)
    truth[count // 2:] = np.resize([1., -1.], count - count // 2)
    observed = matrix @ truth
    curvature = float(np.linalg.norm(matrix, 2) ** 2 * curvature_multiplier)
    tree = P.build_candidate_tree(np.arange(count)[:, None], max_depth=int(config['max_depth']))
    rate, step = float(config['proximal_rate']), [config['quant_step']]
    state = P.project_rd_tree(np.zeros((count, 1)), tree, step, rate)
    latent = np.zeros(count)
    history = []
    extra_evaluations, projection_calls, gradient_evaluations = 0, 0, 0
    proposal_log = []
    for iteration in range(int(config['proximal_iterations']) + 1):
        field = state.field[:, 0]
        residual = matrix @ field - observed
        distortion = float(.5 * residual @ residual)
        history.append({'iteration': iteration, 'distortion': distortion,
                        'objective': distortion + rate * state.payload_bits,
                        'splits': len(state.split_nodes), 'payload_bits': state.payload_bits})
        if iteration == int(config['proximal_iterations']):
            break
        gradient = matrix.T @ residual
        gradient_evaluations += 1
        if mode == 'rd_line_search':
            def evaluate(field):
                error = matrix @ field[:, 0] - observed
                return float(.5 * error @ error)
            update = P.rd_proposal_step(state, gradient[:, None], tree, [curvature], rate,
                                       config['proposal_scales'], evaluate, distortion)
            state = update.projection
            extra_evaluations += update.distortion_evaluations
            projection_calls += update.projection_calls
            proposal_log.append({'iteration': iteration, 'accepted': update.accepted,
                                 'chosen_scale': update.chosen_scale, 'trials': update.trials})
            if not update.accepted:
                # Same deterministic observations/gradient/scales would repeat.
                # Real minibatch training must NOT stop for one rejected batch.
                break
            continue
        # Shadow mode keeps uncommitted sub-quantum updates. It is a heuristic,
        # not the exact proximal-gradient algorithm and has no descent claim.
        latent = (field if mode == 'hard_prox' else latent) - gradient / curvature
        state = P.project_rd_tree(latent[:, None], tree, step, rate, [curvature])
        projection_calls += 1
    reference = P.project_rd_tree(truth[:, None], tree, step, rate)
    residual = matrix @ reference.field[:, 0] - observed
    reference_objective = float(.5 * residual @ residual + rate * reference.payload_bits)
    values = np.array([entry['objective'] for entry in history])
    increases = np.diff(values)
    maximum_increase = float(increases.max()) if len(increases) else 0.
    if mode in ('hard_prox', 'rd_line_search'):
        assert maximum_increase <= 1e-10
    return {'seed': seed, 'mode': mode, 'curvature_multiplier': curvature_multiplier,
            'curvature': curvature, 'maximum_objective_increase': maximum_increase,
            'initial_objective': float(values[0]), 'final_objective': float(values[-1]),
            'best_objective': float(values.min()), 'known_feasible_reference': reference_objective,
            'gradient_evaluations': gradient_evaluations,
            'extra_distortion_evaluations': extra_evaluations, 'projection_calls': projection_calls,
            'proposal_log': proposal_log, 'history': history}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=ROOT / 'config/idea_rd_tree.yaml')
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    if Path(config['experiment_name']).name != config['experiment_name']:
        raise ValueError('experiment_name must be a directory basename')
    output = ROOT / 'output' / (config['experiment_name'] + '_' + datetime.now().strftime('%Y%m%dT%H%M%S_%f'))
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / 'source_snapshot'
    snapshot.mkdir()
    sources = [Path(__file__).resolve(), ROOT / 'pipeline/ReCon-GS/model/rd_tree_projection.py',
               ROOT / 'tests/test_rd_tree_projection.py', args.config.resolve()]
    manifest = []
    for path in sources:
        shutil.copy2(path, snapshot / path.name)
        manifest.append({'path': str(path.relative_to(ROOT)),
                         'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    (output / 'source_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    count = int(config['points'])
    if count < 8 or count % 8:
        raise ValueError('points must be a positive multiple of eight')
    xyz = np.arange(count, dtype=np.float64)[:, None]
    tree = P.build_candidate_tree(xyz, max_depth=int(config['max_depth']))
    scenarios = {'coherent_small': np.ones(count), 'coherent_large': np.full(count, 8.),
                 'two_regions': np.repeat([1., -1.], count // 2),
                 'hidden_fine': np.tile([1., -1.], count // 2),
                 'localized_detail': np.ones(count)}
    scenarios['localized_detail'][-8:] += np.tile([.5, -.5], 4)
    rows = []
    for name, target in scenarios.items():
        for rate in config['rate_weights']:
            started = time.perf_counter()
            result = P.project_rd_tree(target[:, None], tree, [config['quant_step']], float(rate))
            elapsed = time.perf_counter() - started
            packet = P.encode_projection(result, tree)
            decoded = P.decode_projection(packet, tree)
            assert np.array_equal(decoded, result.field)
            filename = f'{name}_lambda{rate}.rdp'
            (output / filename).write_bytes(packet)
            rows.append({'scenario': name, 'lambda': rate, 'splits': len(result.split_nodes),
                         'payload_bits': result.payload_bits, 'packet_bytes': len(packet),
                         'mse': float(np.mean((decoded[:, 0] - target) ** 2)),
                         'objective': result.objective, 'projection_cpu_ms': elapsed * 1000,
                         'split_nodes': list(result.split_nodes), 'packet': filename})

    optimizations = [coupled_optimization(config, int(seed), float(multiplier), mode)
                     for seed in config['optimization_seeds']
                     for multiplier in config['curvature_multipliers']
                     for mode in config['optimization_modes']]
    report = {'config': config, 'scope': 'Synthetic quadratic observations, not 3DGS rendering.',
              'codec': 'Signed Exp-Golomb + tree flags; shared candidate tree required; fixed header and byte padding reported.',
              'rate_sweep': rows, 'optimizations': optimizations}
    (output / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
    lines = ['# RD tree mathematical experiment', '', report['scope'], '', report['codec'], '',
             '| Scenario | lambda | Splits | Payload bits | Packet bytes | MSE |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    lines += [f"| {r['scenario']} | {r['lambda']} | {r['splits']} | {r['payload_bits']} | {r['packet_bytes']} | {r['mse']:.6g} |"
              for r in rows]
    lines += ['', '| Seed | Curvature multiplier | Mode | Final J | Best J | Feasible reference J | Max increase | Gradients | Extra D evals |',
              '| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    lines += [f"| {r['seed']} | {r['curvature_multiplier']} | {r['mode']} | {r['final_objective']:.6f} | "
              f"{r['best_objective']:.6f} | {r['known_feasible_reference']:.6f} | {r['maximum_objective_increase']:.6f} | "
              f"{r['gradient_evaluations']} | {r['extra_distortion_evaluations']} |"
              for r in optimizations]
    lines += ['', 'The feasible reference uses synthetic ground truth, is NOT available to either optimizer, '
              'and is NOT a global optimum certificate. Shadow accumulation has no monotonicity guarantee.', '',
              'These checks do not establish PSNR gains, realistic renderer curvature, GPU cost, or novelty.']
    (output / 'report.md').write_text('\n'.join(lines) + '\n')
    print(output)
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
