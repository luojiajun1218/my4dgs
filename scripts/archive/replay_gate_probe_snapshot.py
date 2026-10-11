"""Replay the exact saved local candidate plan without running the trainer.

python scripts/archive/replay_gate_probe_snapshot.py SNAPSHOT NEW_OUTPUT_DIRECTORY
Reads fingerprinted external images and frozen GS/motion state. The output must
be new. Fitting/check views, precision choices and budgets come from the snapshot.
"""

import argparse
from dataclasses import asdict
import importlib
import json
from pathlib import Path
import sys

import torch

repo = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo))


def replay(snapshot, output, device='cuda'):
    from libgs.utils.config import to_structured
    from libgs.renderer import RendererConfig
    from scripts.archive.diagnose_binary_gates import file_digest

    package = importlib.import_module('pipeline.ReCon-GS.archive')
    diagnostic = importlib.import_module('pipeline.ReCon-GS.archive.model.gate_diagnostics')
    snapshots = importlib.import_module('pipeline.ReCon-GS.archive.model.gate_probe_snapshot')
    render = importlib.import_module('pipeline.ReCon-GS.archive.gate_diagnostic_render')
    gaussian = importlib.import_module('pipeline.ReCon-GS.archive.model.gaussian')
    snapshot, output = Path(snapshot).resolve(), Path(output).resolve()
    if output.exists() or snapshot.is_relative_to(output):
        raise ValueError('Replay output must be new and outside the source snapshot directory')
    source_digest = file_digest(snapshot)
    packet = torch.load(snapshot, map_location='cpu', weights_only=False)
    motion = snapshots.restore_probe_motion(packet, device)
    config = diagnostic.GateDiagnosticsConfig(**packet['diagnostic_config'])
    config.validate(motion.config)
    data_config, unused = to_structured(packet['data_config'], package.DataConfig)
    if unused:
        raise ValueError(f'Unknown saved data fields: {unused}')
    renderer_config, unused = to_structured(packet['renderer_config'], RendererConfig)
    if unused:
        raise ValueError(f'Unknown saved renderer fields: {unused}')
    state = packet['render_state']
    model = gaussian.GaussianModel(state['max_sh_degree'])
    model.active_sh_degree = state['active_sh_degree']
    model.render_attribute_cache = state['render_attribute_cache']
    for name, value in state['attributes'].items():
        setattr(model, name, torch.nn.Parameter(value.to(device), requires_grad=False))
    objective = render.FrozenImageObjective(model, renderer_config, packet['background'].to(device),
        packet['module_config']['lambda_dssim'], state['old_count'])
    with diagnostic.isolated_rng(device):
        data = package.DataModule(data_config)
        data.setup()
        data.current_frame = packet['frame']
        names = [entry['camera'] for entry in packet['views']]
        loaded = [now for now, _ in data.selective_training_pairs(names, device, 0.)]
        snapshots.verify_snapshot_views(packet, loaded)
        by_name = {view.path.name: view for view in loaded}
        views = [by_name[name] for name in config.camera_names]
        check_views = [by_name[name] for name in config.evaluation_camera_names]
        reference = motion.reference_storage_bytes(packet['first_frame_gs_count'])
        weight = motion.config.storage_loss_weight * motion.storage_scale(packet['step'])
        if motion.stage(packet['step']) not in {'gate_compress', 'gate_refine'}:
            weight = 0.
        output.mkdir(parents=True, exist_ok=False)
        (output / 'source.json').write_text(json.dumps(dict(snapshot=str(snapshot),
            snapshot_sha256=source_digest, frame=packet['frame'], step=packet['step'],
            diagnostic_config=asdict(config), dependencies_verified=True,
            snapshot_purpose=packet.get('purpose', 'training_probe'),
            context='replay_of_pre_update_training_snapshot',
            note='Same saved source state and private candidate plan; finite fitting budget and '
                'restricted uniform descendant precision space. Check cameras remain normal training cameras.'), indent=2),
            encoding='utf-8')
        path = diagnostic.run_probes(motion, frame=packet['frame'], step=packet['step'], context='snapshot_replay',
            views=views, evaluation_views=check_views, image_loss=objective, config=config,
            reference_bytes=reference, rate_weight=weight, output_dir=output,
            samples=packet['samples'], view_plans=packet['view_plans'],
            gradient_components=packet.get('gradient_components'), snapshot_path=snapshot)
    if file_digest(snapshot) != source_digest:
        raise AssertionError('Source snapshot changed during replay')
    print(f'Snapshot replay completed: {path}', flush=True)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('snapshot', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    replay(args.snapshot, args.output, args.device)


if __name__ == '__main__':
    main()
