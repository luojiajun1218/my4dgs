"""Paired read-only render benchmarks of completed hierarchy checkpoints."""

import argparse
import gc
import json
import os
import statistics
import subprocess
import tempfile
from pathlib import Path

import yaml

from benchmark_recongs_native_stream_playback import get_cam00_view, make_pipeline
from run_recongs_native_stream import synchronized_fps


def gpu_process_states():
    result = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
                                      '--format=csv,noheader,nounits'], text=True)
    records = []
    for line in result.splitlines():
        if not line.strip():
            continue
        pid = int(line.split(',', 1)[0])
        if pid == os.getpid():
            continue
        try:
            state = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[0]
        except FileNotFoundError:
            continue
        records.append(dict(pid=pid, state=state, nvidia_record=line,
                            paused=state in {'T', 't'}))
    return records


def other_gpu_processes():
    # SIGSTOP preserves another experiment's CUDA memory, while stopping all
    # of its CPU threads from submitting work. Check again around every sample.
    return [record for record in gpu_process_states() if not record['paused']]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('spatial', type=Path)
    parser.add_argument('cluster', type=Path)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    protocol = yaml.safe_load(args.config.read_text())
    source = Path(__file__).resolve().parents[1]
    records = []
    retained_paused_contexts = gpu_process_states()
    if other_gpu_processes():
        raise RuntimeError('Paired FPS benchmark requires an otherwise idle compute GPU')
    for frame in protocol['checkpoint_frames']:
        with tempfile.TemporaryDirectory(prefix='rigid_tree_pair_fps_') as temporary:
            models, views, quality = {}, {}, {}
            for label, run in [('spatial', args.spatial), ('rigid_cluster', args.cluster)]:
                checkpoint = run / f'ckpt-{frame}-{protocol["checkpoint_step"]}.pth'
                pipeline, torch, _, _ = make_pipeline(run / 'config.yaml', source,
                    checkpoint, Path(temporary) / label)
                module = pipeline.module
                module.eval()
                viewpoint, camera = get_cam00_view(pipeline, frame, module.device)
                if camera != 'cam00.png':
                    raise ValueError(camera)
                expected = json.loads((run / 'frame_summary' / f'frame_{frame:03d}.json')
                                     .read_text())['quantized_evaluation']['test']
                from libgs.metric import psnr, ssim
                with torch.no_grad():
                    image = module.forward(viewpoint)['render'].clamp(0., 1.)
                    measured = dict(psnr=float(psnr(image, viewpoint.image)),
                                    ssim=float(ssim(image, viewpoint.image)),
                                    l1=float(torch.nn.functional.l1_loss(image, viewpoint.image)))
                differences = {key: measured[key] - expected[key] for key in measured}
                if any(abs(value) > protocol['metric_absolute_tolerance']
                       for value in differences.values()):
                    raise ValueError(f'{label} frame {frame}: restored quality differs {differences}')
                quality[label] = dict(restored=measured, restored_minus_original=differences,
                    gs=len(module.gaussians.get_xyz), image_shape=list(viewpoint.image.shape))
                models[label], views[label] = pipeline, viewpoint
                del image, module, pipeline, viewpoint
            samples = dict(spatial=[], rigid_cluster=[])
            with torch.no_grad():
                for trial in range(protocol['paired_rounds']):
                    order = ['spatial', 'rigid_cluster'] if trial % 2 == 0 else ['rigid_cluster', 'spatial']
                    for label in order:
                        foreign = other_gpu_processes()
                        if foreign:
                            raise RuntimeError(f'GPU contention before timing: {foreign}')
                        module = models[label].module
                        result = synchronized_fps(lambda: module.forward(views[label]), torch,
                            module.device, warmup=protocol['warmup_renders'],
                            repeats=protocol['timed_renders'])
                        result['round'] = trial
                        samples[label].append(result)
                        if other_gpu_processes():
                            raise RuntimeError('GPU contention during timing')
                # Aggregate by render time; report spread instead of one noisy sample.
            summary = {label: dict(fps=protocol['timed_renders'] * len(values) /
                sum(value['seconds'] for value in values),
                median_fps=statistics.median(value['fps'] for value in values),
                min_fps=min(value['fps'] for value in values),
                max_fps=max(value['fps'] for value in values)) for label, values in samples.items()}
            records.append(dict(frame=frame, summary=summary, samples=samples,
                                restored_quality=quality,
                                device_name=torch.cuda.get_device_name(module.device)))
            print(json.dumps(dict(frame=frame, summary=summary)), flush=True)
            del module, models, views
            gc.collect()
            torch.cuda.empty_cache()
    means = {label: dict(fps=sum(len(row['samples'][label]) * protocol['timed_renders']
                for row in records) / sum(sample['seconds'] for row in records
                    for sample in row['samples'][label]),
                mean_gs=statistics.fmean(row['restored_quality'][label]['gs'] for row in records))
             for label in ['spatial', 'rigid_cluster']}
    result = dict(spatial=str(args.spatial), rigid_cluster=str(args.cluster),
        retained_paused_gpu_contexts=retained_paused_contexts,
        protocol=protocol, records=records, aggregate=means, no_training_updates=True,
        fps_change_percent=100 * (means['rigid_cluster']['fps'] / means['spatial']['fps'] - 1),
        scope='Materialized-GS module.forward only, cam00, synchronized batch timing; excludes '
              'clamp, metrics, loading, storage decoding, motion update, CPU export and training.',
        limitations=[f'{len(records)} saved frames, not a whole sequence rendering average.',
                     'Samples alternate on the same GPU with no other runnable compute processes. '
                     'Paused processes retain CUDA memory and are recorded separately.',
                     'Checkpoint PSNR/SSIM/L1 are checked against the original final evaluation.'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(dict(aggregate=means, fps_change_percent=result['fps_change_percent'])))


if __name__ == '__main__':
    main()
