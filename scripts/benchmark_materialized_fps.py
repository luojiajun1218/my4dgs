"""Read-only, exclusive-GPU materialized checkpoint timing at fixed resolution."""

import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

import yaml

from benchmark_recongs_native_stream_playback import make_pipeline, get_cam00_view


def check_exclusive_gpu():
    lines = subprocess.check_output(['nvidia-smi',
        '--query-compute-apps=pid,process_name,used_gpu_memory',
        '--format=csv,noheader,nounits'], text=True).splitlines()
    foreign = [line for line in lines if line.strip() and int(line.split(',')[0]) != os.getpid()]
    if foreign:
        raise RuntimeError(f'Exclusive GPU timing found other compute contexts: {foreign}')


def time_renders(function, torch, protocol):
    for _ in range(protocol['warmup_renders']):
        function()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(protocol['timed_renders']):
        function()
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    return dict(seconds=seconds, fps=protocol['timed_renders'] / seconds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run, output = args.run.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    protocol = yaml.safe_load(args.protocol.read_text())
    check_exclusive_gpu()
    records = []
    for frame in protocol['checkpoint_frames']:
        with tempfile.TemporaryDirectory(prefix='materialized_fps_') as temporary:
            checkpoint = run / f'ckpt-{frame}-{protocol["checkpoint_step"]}.pth'
            config_path = run / 'config.yaml'
            if 'evaluation_resolution' in protocol:
                evaluation_config = yaml.safe_load(config_path.read_text())
                evaluation_config['data']['resolution'] = protocol['evaluation_resolution']
                evaluation_config['data']['expected_image_size'] = protocol['image_size']
                config_path = Path(temporary) / 'evaluation_config.yaml'
                config_path.write_text(yaml.safe_dump(evaluation_config, sort_keys=False))
            pipeline, torch, _, _ = make_pipeline(config_path,
                Path(__file__).resolve().parents[1], checkpoint, Path(temporary))
            module = pipeline.module
            module.eval()
            view, camera = get_cam00_view(pipeline, frame, module.device)
            if camera != 'cam00.png' or list(view.image.shape[-2:][::-1]) != protocol['image_size']:
                raise ValueError('Camera or image dimensions differ from timing protocol')
            report_path = run / 'frame_summary' / f'frame_{frame:03d}.json'
            initial_only = protocol.get('quality_reference') == 'initial_checkpoint'
            if initial_only and frame != 0:
                raise ValueError('Initial checkpoint quality mode only accepts frame zero')
            expected = None if initial_only else json.loads(report_path.read_text())
            from libgs.metric import psnr, ssim
            renderer = module.renderer
            with torch.no_grad():
                image = module.forward(view)['render'].clamp(0., 1.)
                quality = dict(psnr=float(psnr(image, view.image)), ssim=float(ssim(image, view.image)))
                reference = expected['quantized_evaluation']['test'] if expected is not None else quality
                if any(abs(quality[key] - reference[key]) > protocol['metric_absolute_tolerance']
                       for key in quality):
                    raise ValueError(f'Restored quality differs at frame {frame}: {quality}, {reference}')
                kwargs = dict(means3D=module.gaussians.get_xyz,
                    means2D=renderer.create_screenspace_points(), shs=None, colors_precomp=None,
                    opacities=module.gaussians.get_opacity, scales=module.gaussians.get_scaling,
                    rotations=module.gaussians.get_rotation, cov3D_precomp=None)
                kwargs = renderer.apply_color(kwargs, view, None)
                rasterizer = renderer.create_rasterizer(view, module.background, 1.)
                functions = dict(forward_clamp=lambda: module.forward(view)['render'].clamp(0., 1.),
                                 rasterizer=lambda: rasterizer(**kwargs))
                samples = {name: [] for name in protocol['scopes']}
                for repeat in range(protocol['rounds']):
                    names = protocol['scopes'] if repeat % 2 == 0 else list(reversed(protocol['scopes']))
                    for name in names:
                        check_exclusive_gpu()
                        samples[name].append(time_renders(functions[name], torch, protocol))
                        check_exclusive_gpu()
            summary = {name: dict(fps=protocol['timed_renders'] * len(values) /
                sum(value['seconds'] for value in values),
                min_fps=min(value['fps'] for value in values),
                max_fps=max(value['fps'] for value in values)) for name, values in samples.items()}
            records.append(dict(frame=frame, gs=len(module.gaussians.get_xyz),
                quality=quality, samples=samples, summary=summary,
                render_attribute_cache=module.gaussians.render_attribute_cache))
            print(json.dumps(dict(frame=frame, summary=summary)), flush=True)
            del functions, kwargs, rasterizer, image, module, renderer, view, pipeline
            gc.collect()
            torch.cuda.empty_cache()
    aggregate = {name: protocol['timed_renders'] * protocol['rounds'] * len(records) /
        sum(sample['seconds'] for row in records for sample in row['samples'][name])
        for name in protocol['scopes']}
    result = dict(run=str(run), protocol=protocol, records=records, aggregate_fps=aggregate,
        gpu=torch.cuda.get_device_name(), pid=os.getpid(), no_training_updates=True,
        scope_semantics=dict(forward_clamp='Synchronized materialized module.forward plus image clamp; '
            'includes property preparation and Python submission, excludes metrics/loading/decoding.',
            rasterizer='Synchronized rasterizer calls with camera and Gaussian arguments prepared once; '
            'excludes attribute activation, screen-space allocation, camera preparation and clamp.'))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(dict(aggregate_fps=aggregate)))


if __name__ == '__main__':
    main()
