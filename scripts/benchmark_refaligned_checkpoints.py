"""Read-only checkpoint FPS measurement using native REF's exact 5/30 scope."""
import argparse
import gc
import json
import tempfile
from pathlib import Path

import yaml

from benchmark_recongs_native_stream_playback import make_pipeline, get_cam00_view
from run_recongs_native_stream import synchronized_fps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    parser.add_argument('--benchmark-config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    protocol = yaml.safe_load(args.benchmark_config.read_text())
    reference = {row['frame']: row for line in Path(protocol['reference_metrics']).read_text().splitlines()
                 if (row := json.loads(line))['frame'] > 0}
    records = []
    for frame in protocol['checkpoint_frames']:
        checkpoint = args.run / f'ckpt-{frame}-200.pth'
        assert checkpoint.exists()
        with tempfile.TemporaryDirectory(prefix=f'refaligned_fps{frame}_', dir='/tmp') as workspace:
            pipeline, torch, _, _ = make_pipeline(args.run / 'config.yaml',
                Path(__file__).resolve().parents[1], checkpoint, Path(workspace))
            module = pipeline.module
            module.eval()
            viewpoint, camera = get_cam00_view(pipeline, frame, module.device)
            assert camera == 'cam00.png' and list(viewpoint.image.shape) == [3, 1199, 1600]
            expected = json.loads((args.run / 'frame_summary' /
                                   f'frame_{frame:03d}.json').read_text())['quantized_evaluation']['test']
            from libgs.metric import psnr, ssim
            with torch.no_grad():
                image = module.forward(viewpoint)['render'].clamp(0., 1.)
                measured = dict(psnr=float(psnr(image, viewpoint.image)),
                                ssim=float(ssim(image, viewpoint.image)),
                                l1=float(torch.nn.functional.l1_loss(image, viewpoint.image)))
                differences = {key: measured[key] - expected[key] for key in measured}
                assert all(abs(value) <= protocol['metric_absolute_tolerance']
                           for value in differences.values()), differences
                benchmark = synchronized_fps(lambda: module.forward(viewpoint), torch, module.device,
                    warmup=protocol['warmup_renders'], repeats=protocol['timed_renders'])
            original = reference[frame]['renderer_benchmark']
            assert original['warmup_renders'] == benchmark['warmup_renders']
            assert original['timed_renders'] == benchmark['timed_renders']
            assert original['scope'] == benchmark['scope']
            records.append(dict(frame=frame, camera=camera, candidate=benchmark,
                                historical_reference=original,
                                restored_test_metrics=measured,
                                restored_minus_original_test=differences,
                                device_name=torch.cuda.get_device_name(module.device),
                                torch_version=torch.__version__))
            del image, viewpoint, module, pipeline
            gc.collect()
            torch.cuda.empty_cache()
    summary = dict(records=records, no_training_updates=True,
                   protocol_scope='cam00 module.forward only, no clamp;5warmups+30synchronized batch calls',
                   limitations=[
                       'Same timing range and warmup; historical REF measurement was taken at another time.',
                       'Not a concurrent controlled hardware benchmark or a whole299 FPS average.',
                       'Checkpoint restores are checked against their original test quality before timing.',
                       'Setup/load/GT/metrics/temp logs are excluded from FPS; no optimizer steps.'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
