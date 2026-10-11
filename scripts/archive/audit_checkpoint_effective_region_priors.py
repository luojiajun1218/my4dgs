"""Read saved CPU poses and distinguish raw region priors from GS motion."""
import argparse
import hashlib
import json
import runpy
from pathlib import Path

import torch
import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(0)
    torch.set_num_threads(2)
    scope = runpy.run_path('tests/test_nested_prototype.py', run_name='saved_effective_prior_audit')
    packet = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    saved = packet['quantized_residual_motion']
    motion = scope['GR'].GatedResidualMotion(scope['Q'].QuantizedResidualConfig(**saved['config']))
    motion.restore(saved)
    gate = motion.gate_state_arrays()
    symbols = motion._dense_symbols(integer=True)
    levels = []
    for level in range(3):
        selected = gate['visited'][level] & (gate['states'][level] != scope['GR'].KEEP)
        index = motion.index[level]
        total = torch.bincount(index, minlength=motion.counts[level])
        exposed = torch.bincount(index[gate['gs_terminal'] != scope['GR'].KEEP],
                                 minlength=motion.counts[level])
        hidden = selected & (total > 0) & (exposed == 0)
        partial = selected & (exposed > 0) & (exposed < total)
        values = symbols[f'region_{level}'].to(motion.base_xyz) * motion.motion_steps_vector
        nonzero = (values != 0).any(-1)
        levels.append(dict(level=level, raw_nonzero_region_count=int(nonzero.sum()),
            reachable_nonKEEP_count=int(selected.sum()),
            reachable_nonKEEP_nonzero_count=int((selected & nonzero).sum()),
            fully_canceled_by_descendant_KEEP_count=int(hidden.sum()),
            fully_canceled_nonzero_count=int((hidden & nonzero).sum()),
            fully_canceled_nonzero_nodes=torch.where(hidden & nonzero)[0].tolist(),
            partially_canceled_count=int(partial.sum()),
            fully_canceled_raw_translation_norm_max=(float(values[hidden, :3].norm(dim=-1).max())
                                                       if hidden.any() else 0.),
            fully_canceled_raw_rotation_vector_norm_max=(float(values[hidden, 3:].norm(dim=-1).max())
                                                         if hidden.any() else 0.)))
    with torch.no_grad():
        xyz, rotation = motion(200)
    diagnostics = yaml.safe_load((args.checkpoint.parent / 'config.yaml').read_text())['module']['quantized_diagnostics']
    measured = scope['GR'].actual_motion_statistics(motion.base_xyz, motion.base_rotation,
        xyz, rotation, motion.extent, diagnostics['translation_extent_thresholds'],
        diagnostics['rotation_radian_thresholds'], gate['gs_terminal'])
    assert measured['keep_position_changed_count'] == 0 and measured['keep_rotation_changed_count'] == 0
    result = dict(checkpoint=str(args.checkpoint), sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        frame=packet['current_frame'], motion_init_decay=saved['config']['motion_init_decay'],
        region_levels=levels, effective_final_GS_motion=measured,
        semantics='Raw quantized parameters and reachable nonKEEP priors are distinguished from final GS motion. Descendant absoluteKEEP can cancel an ancestor region; this records exposure, not root-cause attribution. No data/render/optimization is performed.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(dict(frame=result['frame'], levels=levels,
                         actual_changed_count=measured['any_pose_changed_count'])))


if __name__ == '__main__':
    main()
