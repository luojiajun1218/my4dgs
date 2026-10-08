"""Summarize node-level evidence, keeping online and checkpoint contexts separate.

python scripts/summarize_gate_diagnostics.py RUN OUTPUT_JSON
Reads diagnostic records only; refuses to overwrite the explicit output file.
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics


def summarize(run):
    run = Path(run)
    directory = run / 'gate_diagnostics' if (run / 'gate_diagnostics').is_dir() else run
    rows = []
    joint_nodes, joint_candidates = [], []
    for path in sorted(directory.glob('frame_*_step_*.jsonl')):
        for line_number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
            record = json.loads(line)
            if record.get('schema') == 'joint_precision_candidates_v1':
                identity = dict(file=str(path.relative_to(run)), line=line_number,
                    frame=record['frame'], step=record['step'], context=record['context'],
                    level=record['level'], node=record['node'])
                joint_nodes.append(dict(**identity, original_open=record['original']['gate_open'],
                    selected_by_fitting_rate_score=record['selected_by_fitting_rate_score'],
                    selected_by_check_rate_score=record['selected_by_check_rate_score'],
                    best_stop=record['best_stop'], best_refine=record['best_refine'],
                    best_refine_minus_stop_gain=record['best_refine_minus_stop_gain'],
                    precision_space=record['precision_space']))
                for candidate in record['candidates']:
                    joint_candidates.append(dict(**identity, **candidate['spec'],
                        image_gain=candidate['image_gain'], gain=candidate['gain'],
                        evaluation_gain=candidate['evaluation_gain'], byte_delta=candidate['byte_delta'],
                        before=candidate['before'], after=candidate['after'],
                        per_view_gains=candidate['per_view_gains'], fit_seconds=candidate['fit_seconds']))
                continue
            for comparison in record['comparisons']:
                stop, refine = comparison['arms']['stop'], comparison['arms']['refine']
                gain = comparison['gain']
                selected = record['original']['gate_open']
                rows.append(dict(file=str(path.relative_to(run)), line=line_number,
                    frame=record['frame'], step=record['step'], context=record['context'],
                    stage=record['stage'], level=record['level'], node=record['node'],
                    original_open=selected, precision_policy=comparison['precision_policy'],
                    gain=gain, distortion_gain=comparison['distortion_gain'],
                    byte_delta=comparison['byte_delta'], rate_weight=comparison['rate_weight'],
                    reference_bytes=comparison['reference_bytes'],
                    gate_agrees=None if gain == 0 else selected == (gain > 0),
                    stop_fit_improvement=stop['before']['image_loss'] - stop['after']['image_loss'],
                    refine_fit_improvement=refine['before']['image_loss'] - refine['after']['image_loss'],
                    stop_loss=stop['after']['image_loss'], refine_loss=refine['after']['image_loss'],
                    fit_seconds=stop['fit_seconds'] + refine['fit_seconds'],
                    total_seconds=stop['total_seconds'] + refine['total_seconds']))
    if not rows and not joint_nodes:
        raise ValueError(f'No completed gate comparisons in {directory}')
    groups = defaultdict(list)
    for row in rows:
        groups[(row['context'], row['frame'], row['step'], row['precision_policy'])].append(row)
    aggregates = []
    for (context, frame, step, policy), values in sorted(groups.items()):
        agreement = [value['gate_agrees'] for value in values if value['gate_agrees'] is not None]
        aggregates.append(dict(context=context, frame=frame, step=step, precision_policy=policy,
            sampled_nodes=len(values), gate_agreement_fraction=statistics.fmean(agreement) if agreement else None,
            closed_but_refine_supported=sum(not value['original_open'] and value['gain'] > 0 for value in values),
            open_but_stop_supported=sum(value['original_open'] and value['gain'] < 0 for value in values),
            mean_stop_fit_improvement=statistics.fmean(value['stop_fit_improvement'] for value in values),
            mean_refine_fit_improvement=statistics.fmean(value['refine_fit_improvement'] for value in values),
            mean_byte_delta=statistics.fmean(value['byte_delta'] for value in values),
            fit_seconds=sum(value['fit_seconds'] for value in values)))
    return dict(run=str(run), aggregates=aggregates, rows=rows,
        joint_nodes=joint_nodes, joint_candidates=joint_candidates,
        interpretation='G compares fitted candidates using image loss and actual byte delta. '
            'Samples are stratified, not a population-weighted estimate. Equal G is reported as a tie. '
            'Online gradients include the main training objective; checkpoint gradients are unavailable. '
            'Finite extra fitting budgets do not establish convergence; final-state alternatives cannot '
            'by themselves establish gate-learning bias. Joint scores are local restricted '
            'parent/uniform-descendant comparisons, not scene-wide gate accuracy or a whole-tree optimum. '
            'Check cameras are normal training cameras excluded only from extra candidate fitting; '
            'region gains cannot be added because rendering and frame padding couple them.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite {args.output}')
    result = summarize(args.run)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(f"Summarized {len(result['rows'])} stop/refine comparisons and "
          f"{len(result['joint_nodes'])} joint nodes: {args.output}")


if __name__ == '__main__':
    main()
