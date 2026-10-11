"""Launch the configured diagnostic run, or inspect an existing checkpoint.

Training: python scripts/archive/diagnose_binary_gates.py --config config/NEW_RUN.yaml
Read-only checkpoint fit: add --checkpoint /path/to/ckpt-FRAME-STEP.pth
Both modes create a separate output run; source checkpoints are never written.
"""

import argparse
from dataclasses import asdict
from datetime import datetime
import hashlib
import importlib
import json
from pathlib import Path
import sys
import warnings

import torch
from omegaconf import OmegaConf

repo = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo))


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path, config_class):
    """Resolve YAML to native containers before constructing pipeline dataclasses."""
    from libgs.utils.config import to_structured
    values = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    with warnings.catch_warnings():
        # libgs tries to call typing.Optional itself, then keeps the valid string.
        warnings.filterwarnings(
            'ignore', category=UserWarning, module=r'libgs\.utils\.config',
            message=r"(?:Try|Failed) to cast '.*' to typing\.Optional\[(?:pathlib\.Path|str)\]$",
        )
        config, unused = to_structured(values, config_class)
    if unused:
        raise ValueError(f'Unknown configuration fields: {unused}')
    return config


def checkpoint_diagnostics(config, checkpoint):
    from libgs.utils.config import to_yaml

    package = importlib.import_module('pipeline.ReCon-GS.archive')
    diagnostic = importlib.import_module('pipeline.ReCon-GS.archive.model.gate_diagnostics')
    render = importlib.import_module('pipeline.ReCon-GS.archive.gate_diagnostic_render')
    sab = importlib.import_module('pipeline.ReCon-GS.archive.model.storage_aware_binary')
    gaussian = importlib.import_module('pipeline.ReCon-GS.archive.model.gaussian')
    if config.module.motion_mode != 'storage_aware_binary':
        raise ValueError('Expected storage_aware_binary config')
    cfg = config.module.gate_diagnostics
    cfg.validate(config.module.storage_aware_binary)
    if not cfg.camera_names:
        raise ValueError('Checkpoint probes require explicit training camera_names')
    if not torch.cuda.is_available():
        raise RuntimeError('Gaussian image diagnostics require CUDA')
    checkpoint = checkpoint.resolve()
    root = (Path(config.output_dir) / config.experiment_name).resolve()
    if checkpoint.is_relative_to(root):
        raise ValueError('Use a new experiment_name outside the historical checkpoint run')
    source_digest = file_digest(checkpoint)
    packet = torch.load(checkpoint, map_location='cuda', weights_only=False)
    state = packet['quantized_residual_motion']
    motion = sab.StorageAwareBinaryMotion(config.module.storage_aware_binary)
    motion.restore(state)
    frame, step = int(packet['current_frame']), int(motion.last_step)
    if frame < 1 or step < 1:
        raise ValueError('Diagnostic checkpoint must contain dynamic-frame motion')
    frame_state = packet['quantized_frame_state']
    if motion.committed and not frame_state.get('finalized', False):
        raise ValueError('Use a motion-learning checkpoint or a finalized checkpoint; '
                         'intermediate newborn optimization is not supported')
    context = 'checkpoint_final' if frame_state.get('finalized', False) else 'checkpoint_motion_phase'
    model = gaussian.GaussianModel(config.module.sh_degree)
    slots = dict(_xyz=1, _features_dc=2, _features_rest=3, _scaling=4, _rotation=5, _opacity=6)
    model.active_sh_degree = int(packet['gaussians'][0])
    for attribute, slot in slots.items():
        setattr(model, attribute, torch.nn.Parameter(packet['gaussians'][slot].detach(), requires_grad=False))
    old_count = len(motion.base_xyz)
    if context == 'checkpoint_final':
        with torch.no_grad():
            xyz, rotation = motion(step)
            torch.testing.assert_close(xyz, model._xyz[:old_count], rtol=0, atol=1e-6)
            torch.testing.assert_close(torch.nn.functional.normalize(rotation, dim=-1),
                torch.nn.functional.normalize(model._rotation[:old_count], dim=-1), rtol=1e-5, atol=1e-6)
    background = (torch.ones if config.data.white_background else torch.zeros)(3, device='cuda')
    objective = render.FrozenImageObjective(model, config.module.renderer, background,
                                           config.module.lambda_dssim, old_count)
    data = package.DataModule(config.data)
    with diagnostic.isolated_rng('cuda'):
        data.setup()
        data.current_frame = frame
        views = [now for now, _ in data.selective_training_pairs(cfg.camera_names, 'cuda', 0.)]
        # Reject validation views even when a custom split uses a shared scene dataset.
        if config.data.training_camera_names is not None and not set(cfg.camera_names).issubset(
                config.data.training_camera_names):
            raise ValueError('Diagnostic fitting cameras must belong to the training split')
        output = root / ('checkpoint_' + datetime.now().strftime('%Y%m%dT%H%M%S_%f'))
        output.mkdir(parents=True, exist_ok=False)
        (output / 'config.yaml').write_text(to_yaml(config), encoding='utf-8')
        metadata = dict(checkpoint=str(checkpoint), checkpoint_sha256=source_digest,
            frame=frame, step=step, context=context, diagnostic=asdict(cfg),
            source_gradients='unavailable_in_checkpoint; candidate gradients are separately labeled',
            limitations=['Frozen newborn attributes and topology; not a whole-frame retraining.',
                        'Fixed extra fitting budget does not establish candidate convergence.',
                        'Final-state alternatives do not diagnose earlier gate-learning bias.'])
        (output / 'source.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        samples = diagnostic.sample_nodes(motion, step, frame, cfg)
        diagnostic.write_trace(output / 'trace.jsonl', frame=frame, step=step, context=context,
            rows=[dict(stage=motion.stage(step), sample=sample,
                gradient_source='unavailable_in_checkpoint',
                **diagnostic.node_trace(motion, step, sample['level'], sample['node'])) for sample in samples])
        weight = motion.config.storage_loss_weight * motion.storage_scale(step)
        if motion.stage(step) == 'dense_fit':
            weight = 0.
        path = diagnostic.run_probes(motion, frame=frame, step=step, context=context,
            views=views, image_loss=objective, config=cfg,
            reference_bytes=motion.reference_storage_bytes(frame_state['first_frame_gs_count']),
            rate_weight=weight, output_dir=output,
            birth_bytes=sab.birth_ply_size(len(model._xyz) - old_count, config.module.sh_degree))
    if file_digest(checkpoint) != source_digest:
        raise AssertionError('Source checkpoint changed during diagnosis')
    print(f'Checkpoint diagnostics: {path}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path)
    args = parser.parse_args()
    from absl import flags
    if not flags.FLAGS.is_parsed():
        flags.FLAGS([sys.argv[0]])
    package = importlib.import_module('pipeline.ReCon-GS.archive')
    config = load_config(args.config, package.Config)
    config.module.gate_diagnostics.validate(config.module.storage_aware_binary)
    if args.checkpoint:
        checkpoint_diagnostics(config, args.checkpoint)
    else:
        if config.mode != 'train' or config.module.motion_mode != 'storage_aware_binary':
            raise ValueError('This entry requires train mode and storage_aware_binary motion')
        if not config.module.gate_diagnostics.enabled:
            raise ValueError('Enable gate_diagnostics explicitly for this diagnostic training entry')
        from absl import logging
        from libgs.utils.git import get_git_commit_hash
        logging.set_verbosity(logging.INFO)
        pipeline = package.Pipeline(config)
        logging.get_absl_handler().use_absl_log_file('logging', pipeline.output_dir)
        logging.info('Gate diagnostic run, source commit %s', get_git_commit_hash())
        print(f'Diagnostic training output: {pipeline.output_dir}', flush=True)
        pipeline.run()


if __name__ == '__main__':
    raise SystemExit('Archived training launcher; use main.py with the rigid-tree config')
