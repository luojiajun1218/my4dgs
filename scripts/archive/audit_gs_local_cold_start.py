"""Inspect sampled local gradients and read-only fine SHARE/SPLIT pose alternatives."""
import argparse
import json
import runpy
import statistics
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    parser.add_argument('--endpoint', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(0)
    samples = []
    for frame in range(1, args.endpoint + 1):
        control = json.loads((args.run / 'frame_summary' /
                              f'control_{frame:03d}.json').read_text())
        assert {row['step'] for row in control['gradients']} == {1, 30, 80, 133}
        samples.extend(control['gradients'])
    scope = runpy.run_path('tests/test_nested_prototype.py', run_name='cold_start_audit')
    gr, q = scope['GR'], scope['Q']
    records = []
    for frame in (5, 24, 49, 74, 99, 124, 149, 174, 199, 224, 249, 274, 299):
        if frame > args.endpoint:
            continue
        saved = torch.load(args.run / f'ckpt-{frame}-200.pth', map_location='cpu',
                           weights_only=False)['quantized_residual_motion']
        motion = gr.GatedResidualMotion(q.QuantizedResidualConfig(**saved['config']))
        motion.restore(saved)
        with torch.no_grad():
            selection = motion.gate_state_arrays()
            mask = selection['visited'][2] & (selection['states'][2] == gr.SHARE)
            states = [value.clone() for value in selection['states']]
            states[2][mask] = gr.SPLIT
            dense = motion._dense_symbols()
            steps = motion.motion_steps_vector
            regions = [dense[f'region_{level}'] * steps for level in range(3)]
            local = dense['gs_local'] * steps
            original = motion(200)
            alternate = gr._motion(motion.base_xyz, motion.base_rotation,
                                   motion.active_layout, regions, local, states)
            diffs = [float((a - b).abs().max()) for a, b in zip(original, alternate)]
            for a, b in zip(original, alternate):
                torch.testing.assert_close(a, b, rtol=0, atol=1e-6)
            records.append(dict(frame=frame,
                local_translation_nonzero_elements=int(torch.count_nonzero(motion.gs_translation)),
                local_rotation_nonzero_elements=int(torch.count_nonzero(motion.gs_rotation)),
                fine_reachable_SHARE_nodes=int(mask.sum()),
                fine_reachable_SPLIT_nodes=int((selection['visited'][2] &
                                                (selection['states'][2] == gr.SPLIT)).sum()),
                counterfactual_promote_all_reachable_fine_SHARE_to_SPLIT_position_max_abs_difference=diffs[0],
                rotation_max_abs_difference=diffs[1]))
    result = dict(frames=[1, args.endpoint], combined_gradient_sample_count=len(samples),
        samples_with_nonzero_GS_translation_gradient=sum(
            row['gradients']['gs_translation']['nonzero_rows'] > 0 for row in samples),
        samples_with_nonzero_GS_rotation_gradient=sum(
            row['gradients']['gs_rotation']['nonzero_rows'] > 0 for row in samples),
        samples_with_fine_SPLIT=sum(
            row['activity']['levels'][2]['counts']['SPLIT'] > 0 for row in samples),
        median_fine_gate_l2_by_step={str(step): statistics.median(
            row['gradients']['gate_2']['l2'] for row in samples if row['step'] == step)
            for step in (1, 30, 80, 133)}, checkpoint_records=records,
        source_evidence=dict(GS_initialization='pipeline/ReCon-GS/archive/model/gated_residual.py:305',
            hard_selected_motion_jacobian='pipeline/ReCon-GS/archive/model/gated_residual.py:618',
            KEEP_SHARE_only_auxiliary='pipeline/ReCon-GS/archive/model/gated_residual.py:632',
            detached_GS_local_in_auxiliary='pipeline/ReCon-GS/archive/model/gated_residual.py:653'),
        limits='CPU numerical pose counterfactual; no image renders or updates. Logged gradients combine main image, auxiliary, storage and GS activity losses. Does not prove the complete test deficit has one cause.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ('checkpoint_records', 'source_evidence')}))


if __name__ == '__main__':
    main()
