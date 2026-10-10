"""Time strict sequential decoding plus one render per frame on an idle GPU."""

import argparse
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import time

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.benchmark_materialized_fps import check_exclusive_gpu
from scripts.benchmark_recongs_native_stream_playback import get_cam00_view

CC = importlib.import_module('pipeline.ReCon-GS.model.common_contrast')
SAB = importlib.import_module('pipeline.ReCon-GS.model.storage_aware_binary')
Q = importlib.import_module('pipeline.ReCon-GS.model.residual_motion')


def install_raw(model, raw):
    """Install decoder-owned tensors without constructing a training optimizer."""
    for name, attribute in zip(Q.RAW_FIELDS, Q.RAW_ATTRIBUTES):
        setattr(model, attribute, torch.nn.Parameter(raw[name], requires_grad=False))
    model._render_cache = {}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--initial-model', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA access is required for verified playback timing')
    check_exclusive_gpu()
    config = yaml.safe_load((args.run / 'config.yaml').read_text())
    protocol = yaml.safe_load(args.protocol.read_text())
    frames = int(protocol['frames'])
    rounds = int(protocol['rounds'])
    quality_frames = set(protocol['quality_frames'])
    if frames < 1 or rounds < 1 or not quality_frames.issubset(range(1, frames + 1)):
        raise ValueError('Invalid playback frame or round selection')
    from libgs.metric import psnr, ssim
    from libgs.renderer import Renderer, RendererConfig
    import libgs.data.dataset.base as dataset_base
    data = importlib.import_module('pipeline.ReCon-GS.data')
    GaussianModel = importlib.import_module('pipeline.ReCon-GS.model.gaussian').GaussianModel

    # Playback reads views in the main process; libgs' warning assumes a worker.
    dataset_base.warning = lambda: None
    datamodule = data.DataModule(data.DataConfig(**config['data']))
    datamodule.setup()
    pipeline = SimpleNamespace(datamodule=datamodule)
    camera, name = get_cam00_view(pipeline, 1, 'cuda')
    if name != 'cam00.png' or list(camera.image.shape[-2:][::-1]) != protocol['image_size']:
        raise ValueError('Camera or image size differs from protocol')
    views = {}
    for frame in quality_frames:
        view, view_name = get_cam00_view(pipeline, frame, 'cuda')
        if view_name != name or any(not torch.equal(getattr(camera, key), getattr(view, key))
                                  for key in ('world_view_transform', 'full_proj_transform', 'camera_center')):
            raise ValueError('This protocol requires a fixed camera throughout playback')
        if (view.fovx, view.fovy, view.image.shape) != (camera.fovx, camera.fovy, camera.image.shape):
            raise ValueError('Playback camera intrinsics differ')
        views[frame] = view
    with np.load(args.initial_model, allow_pickle=False) as archive:
        initial = {name: torch.from_numpy(archive[name].copy()).cuda() for name in Q.RAW_FIELDS}
    metadata = json.loads((args.initial_model.parent / 'initial_model_export.json').read_text())
    degree = int(metadata['active_sh_degree'])
    if (degree + 1) ** 2 - 1 != initial['sh_rest'].shape[1]:
        raise ValueError('Initial SH basis and active degree disagree')
    model = GaussianModel(degree)
    model.active_sh_degree = degree
    model.render_attribute_cache = bool(protocol['render_attribute_cache'])
    renderer = Renderer(RendererConfig(**config['module']['renderer']), model)
    background = torch.tensor([1., 1., 1.] if config['data']['white_background'] else [0., 0., 0.], device='cuda')
    motion_config = CC.CommonContrastConfig(**config['module']['common_contrast_binary'])
    storage = args.run / 'frame_storage' / 'common_contrast_binary'
    checkpoint = torch.load(args.run / f'ckpt-{frames}-{config["trainer"]["num_incr_steps"]}.pth',
                            map_location='cpu', weights_only=False)['gaussians']
    expected = dict(xyz=checkpoint[1].cuda(), sh_dc=checkpoint[2].cuda(),
                    sh_rest=checkpoint[3].cuda(), logscale=checkpoint[4].cuda(),
                    rotation=checkpoint[5].cuda(), opacity=checkpoint[6].cuda())
    passes = []
    with torch.no_grad():
        install_raw(model, initial)
        for _ in range(protocol['warmup_renders']):
            renderer(camera, background)['render'].clamp(0., 1.)
        torch.cuda.synchronize()
        for repeat in range(rounds):
            check_exclusive_gpu()
            decoder = SAB.decoder_for_binary_stream(storage / 'motion_001.bin', motion_config)
            raw = {name: value.clone() for name, value in initial.items()}
            rows = []
            for frame in range(1, frames + 1):
                motion = storage / f'motion_{frame:03d}.bin'
                birth = storage / f'incremental_{frame:03d}.bin'
                if not birth.exists():
                    birth = storage / f'incremental_{frame:03d}.ply'
                appearance = storage / f'appearance_{frame:03d}.bin'
                opacity = storage / f'opacity_{frame:03d}.bin'
                torch.cuda.synchronize()
                started = time.perf_counter()
                raw, _ = decoder.advance(motion, birth if birth.exists() else None, raw,
                    appearance_path=appearance if appearance.exists() else None,
                    opacity_path=opacity if opacity.exists() else None)
                torch.cuda.synchronize()
                decoded = time.perf_counter()
                install_raw(model, raw)
                image = renderer(camera, background)['render'].clamp(0., 1.)
                torch.cuda.synchronize()
                completed = time.perf_counter()
                row = dict(frame=frame, decoder_seconds=decoded - started,
                           install_and_render_seconds=completed - decoded,
                           playback_seconds=completed - started, gs=len(raw['xyz']))
                if frame in quality_frames:
                    quality = dict(psnr=float(psnr(image, views[frame].image)),
                                   ssim=float(ssim(image, views[frame].image)))
                    reference = json.loads((args.run / 'frame_summary' / f'frame_{frame:03d}.json').read_text())['quantized_evaluation']['test']
                    differences = {key: quality[key] - reference[key] for key in quality}
                    if any(abs(value) > protocol['metric_absolute_tolerance'] for value in differences.values()):
                        raise ValueError(f'Playback quality differs: frame {frame}, {differences}')
                    row.update(quality=quality, quality_difference=differences)
                rows.append(row)
            exact = {name: torch.equal(raw[name], expected[name]) for name in Q.RAW_FIELDS}
            if not all(exact.values()):
                raise AssertionError(f'Final playback state differs: {exact}')
            check_exclusive_gpu()
            seconds = sum(row['playback_seconds'] for row in rows)
            passes.append(dict(repeat=repeat, frames=rows, seconds=seconds,
                               playback_fps=frames / seconds, final_fields_exact=exact))
            print(json.dumps(dict(repeat=repeat, playback_fps=frames / seconds)), flush=True)
    result = dict(schema='common_contrast_playback_v1', run=str(args.run.resolve()),
                  bootstrap=str(args.initial_model.resolve()), protocol=protocol,
                  passes=passes, gpu=torch.cuda.get_device_name(),
                  playback_fps=frames * rounds / sum(row['seconds'] for row in passes),
                  scope='Packet reads, strict integrity hashes, deletion, topology, motion, '
                        'birth/color/opacity reconstruction, render attribute installation and '
                        'one synchronized Renderer.forward plus clamp per dynamic frame. '
                        'Excludes initial model and camera loading, quality metrics and endpoint '
                        'checkpoint comparisons. Fixed cam00, local filesystem; later passes may '
                        'benefit from OS file cache. No training optimizer or hidden dynamic reference.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
