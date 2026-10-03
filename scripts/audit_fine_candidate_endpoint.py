"""Read-only numerical comparison of closed local candidates at replay endpoints."""
import argparse
import json
import runpy
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('control', type=Path)
    parser.add_argument('fitted', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frame', type=int, default=15)
    args = parser.parse_args()
    torch.set_num_threads(2)
    scope = runpy.run_path('tests/test_nested_prototype.py', run_name='fine_endpoint_audit')
    gr, q = scope['GR'], scope['Q']
    records = []
    for name, run in (('control', args.control), ('fitted', args.fitted)):
        packet = torch.load(run / f'ckpt-{args.frame}-200.pth', map_location='cpu', weights_only=False)
        saved = packet['quantized_residual_motion']
        assert packet['current_frame'] == args.frame and packet['global_step'] == 200
        assert saved['last_step'] == 133 and saved['committed']
        motion = gr.GatedResidualMotion(q.QuantizedResidualConfig(**saved['config']))
        motion.restore(saved)
        with torch.no_grad():
            selection = motion.gate_state_arrays()
            states = [value.clone() for value in selection['states']]
            eligible = selection['visited'][2] & (states[2] == gr.SHARE)
            states[2][eligible] = gr.SPLIT
            dense = motion._dense_symbols()
            steps = motion.motion_steps_vector
            regions = [dense[f'region_{level}'] * steps for level in range(3)]
            local = dense['gs_local'] * steps
            hard = motion(200)
            alternate = gr._motion(motion.base_xyz, motion.base_rotation,
                                   motion.active_layout, regions, local, states)
            assert all(bool(value.isfinite().all()) for value in (*hard, *alternate, local))
            diff = (alternate[0] - hard[0]).norm(dim=-1)
            records.append(dict(name=name, frame=args.frame, all_finite=True,
                fine_reachable_SHARE_nodes=int(eligible.sum()),
                quantized_local_nonzero_rows=int((local != 0).any(-1).sum()),
                raw_local_nonzero_rows=int(((motion.gs_translation != 0).any(-1) |
                                            (motion.gs_rotation != 0).any(-1)).sum()),
                closed_SHARE_to_SPLIT_position_difference_max=float(diff.max()),
                closed_SHARE_to_SPLIT_position_difference_mean=float(diff.mean()),
                closed_SHARE_to_SPLIT_position_changed_rows=int((diff > 1e-6).sum()),
                closed_SHARE_to_SPLIT_rotation_max_abs=float((alternate[1] - hard[1]).abs().max())))
    result = dict(records=records, image_renders=0, parameter_updates=0,
        candidate_nonidentity_present=records[1]['closed_SHARE_to_SPLIT_position_changed_rows'] > 0 or
            records[1]['closed_SHARE_to_SPLIT_rotation_max_abs'] > 1e-6,
        limits='CPU pose difference proves a distinct trained candidate, not its image benefit or the sole cause of the whole-sequence deficit.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == '__main__':
    main()
