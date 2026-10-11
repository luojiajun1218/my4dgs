"""Strict sequential decode, actual file accounting and separate render time."""

import argparse
import importlib
import json
from pathlib import Path
import sys
import time
from types import ModuleType

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
# The decoder needs numerical modules, not the training GUI socket. Match the
# numerical test entry point so importing an audit does not open a listener.
for name, directory in [('pipeline', ROOT / 'pipeline'),
                        ('pipeline.ReCon-GS', ROOT / 'pipeline/ReCon-GS')]:
    if name not in sys.modules:
        package = ModuleType(name)
        package.__path__ = [str(directory)]
        sys.modules[name] = package
CC = importlib.import_module('pipeline.ReCon-GS.archive.model.common_contrast')
SAB = importlib.import_module('pipeline.ReCon-GS.archive.model.storage_aware_binary')
Q = importlib.import_module('pipeline.ReCon-GS.archive.model.residual_motion')
GAUSS = importlib.import_module('pipeline.ReCon-GS.archive.model.gaussian')
DATA = importlib.import_module('pipeline.ReCon-GS.data')
from libgs.renderer import Renderer, RendererConfig


def install(raw, sh_degree):
    model = GAUSS.GaussianModel(sh_degree)
    for name, attr in zip(Q.RAW_FIELDS, Q.RAW_ATTRIBUTES):
        setattr(model, attr, torch.nn.Parameter(raw[name].clone(), requires_grad=False))
    model.active_sh_degree = sh_degree
    return model


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--frames', type=int, default=5)
    parser.add_argument('--output', type=Path, help='New audit directory outside the source run')
    args = parser.parse_args()
    run = args.run.resolve()
    target = args.output.resolve() if args.output else run
    if args.output:
        target.mkdir(parents=True, exist_ok=False)
    cfg = yaml.safe_load((run / 'config.yaml').read_text())
    config = CC.CommonContrastConfig(**cfg['module']['common_contrast_binary'])
    device = torch.device('cuda')
    steps = int(cfg['trainer']['num_incr_steps'])
    checkpoint = torch.load(run / f'ckpt-1-{steps}.pth', map_location='cpu', weights_only=False)
    initial = {name: value.to(device) for name, value in
               checkpoint['quantized_frame_state']['previous_reference'].items()}
    bootstrap = target / 'deployment_bootstrap'
    bootstrap.mkdir(exist_ok=True)
    first = bootstrap / 'first_frame.ply'
    install(initial, cfg['module']['sh_degree']).save_ply(first)
    raw = SAB._load_birth_ply(first, device)
    if Q.reference_digest(raw) != Q.reference_digest(initial):
        raise AssertionError('First-frame PLY does not exactly reconstruct shared raw fields')
    # Retain only decoder-relevant settings; this file is counted in deployment.
    wire_config = dict(sh_degree=cfg['module']['sh_degree'], codec=config.__dict__)
    config_path = bootstrap / 'codec_config.yaml'
    config_path.write_text(yaml.safe_dump(wire_config, sort_keys=True), encoding='utf-8')
    decoder = CC.CommonContrastDecoder(config)
    data_cfg = DATA.DataConfig(root=cfg['data']['root'], resolution=cfg['data']['resolution'],
        split_train_test=True, shuffle=False, extra_dataset_kwargs=dict(num_frames=args.frames + 1),
        extra_dataloader_kwargs=dict(num_workers=2, pin_memory=True))
    data = DATA.DataModule(data_cfg)
    data.setup()
    rows = []
    storage = run / 'frame_storage' / 'common_contrast_binary'
    for frame in range(1, args.frames + 1):
        motion = storage / f'motion_{frame:03d}.bin'
        birth = storage / f'incremental_{frame:03d}.ply'
        header = CC.CommonContrastMotion.read_header(motion)
        torch.cuda.synchronize()
        started = time.perf_counter()
        raw, info = decoder.advance(motion, birth if birth.exists() else None, raw)
        torch.cuda.synchronize()
        decode_seconds = time.perf_counter() - started
        data.current_frame = frame
        view = next(iter(data.test_dataloader())).to(device)
        if view.path.name != 'cam00.png':
            raise AssertionError('Expected cam00 as the unique test camera')
        model = install(raw, cfg['module']['sh_degree'])
        renderer = Renderer(RendererConfig(**cfg['module']['renderer']), model)
        bg = raw['xyz'].new_zeros(3)
        renderer(view, bg)  # warmup; excluded from timed renders
        elapsed = []
        for _ in range(5):
            torch.cuda.synchronize()
            started = time.perf_counter()
            renderer(view, bg)
            torch.cuda.synchronize()
            elapsed.append(time.perf_counter() - started)
        render_seconds = sum(elapsed) / len(elapsed)
        rows.append(dict(frame=frame, decode_seconds=decode_seconds, render_seconds=render_seconds,
                         motion_bytes=motion.stat().st_size,
                         birth_bytes=birth.stat().st_size if birth.exists() else 0,
                         structure_bytes=header['structure_edit_bytes'],
                         info=info))
        print(json.dumps(rows[-1]), flush=True)
    final = torch.load(run / f'ckpt-{args.frames}-{steps}.pth', map_location=device, weights_only=False)['gaussians']
    expected = dict(xyz=final[1], sh_dc=final[2], sh_rest=final[3],
                    logscale=final[4], rotation=final[5], opacity=final[6])
    fields_exact = {name: torch.equal(raw[name], expected[name]) for name in Q.RAW_FIELDS}
    if not all(fields_exact.values()):
        raise AssertionError(f'Final decoded/checkpoint fields differ: {fields_exact}')
    dynamic_bytes = sum(row['motion_bytes'] + row['birth_bytes'] for row in rows)
    bootstrap_bytes = first.stat().st_size + config_path.stat().st_size
    result = dict(run=str(run), frames=rows, final_fields_exact=fields_exact,
                  final_digest=Q.reference_digest(raw), dynamic_bytes=dynamic_bytes,
                  first_frame_bytes=first.stat().st_size, configuration_bytes=config_path.stat().st_size,
                  bootstrap_bytes=bootstrap_bytes, full_sequence_bytes=bootstrap_bytes + dynamic_bytes,
                  structure_bytes=sum(row['structure_bytes'] for row in rows),
                  structure_fraction_dynamic=sum(row['structure_bytes'] for row in rows) / dynamic_bytes,
                  timing_semantics='Decode includes motion file read, synchronized deletion/tree edit, '
                  'motion application, hashes, birth PLY read and append. Render is five synchronized '
                  'warm renders of one cam00 view, excludes decode, data loading, model install and metrics. '
                  'Bootstrap/data loading are not timed. These are separate timings, not end-to-end playback FPS.')
    (target / 'direct_sequence_audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
