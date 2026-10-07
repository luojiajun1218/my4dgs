"""Restore initial checkpoints and verify cache-independent same-camera quality."""

import argparse
import gc
import hashlib
import json
from pathlib import Path
import statistics
import tempfile

import yaml

from benchmark_recongs_native_stream_playback import make_pipeline


def tensor_digest(value):
    array = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    protocol = yaml.safe_load(args.protocol.read_text())
    ground_truth = {}
    records = []
    for step in protocol['checkpoint_steps']:
        with tempfile.TemporaryDirectory(prefix='initial_quality_') as temporary:
            configuration = yaml.safe_load((args.run / 'config.yaml').read_text())
            configuration['data']['resolution'] = protocol['resolution']
            configuration['data']['expected_image_size'] = protocol['image_size']
            config_path = Path(temporary) / 'evaluation.yaml'
            config_path.write_text(yaml.safe_dump(configuration, sort_keys=False))
            pipeline, torch, _, _ = make_pipeline(config_path,
                Path(__file__).resolve().parents[1], args.run / f'ckpt-0-{step}.pth',
                Path(temporary) / 'pipeline')
            module = pipeline.module
            model = module.gaussians
            module.eval()
            previous_noise = getattr(model, 'noise_scale', None)
            model.noise_scale = None
            model.delta_xyz = model.delta_rot = None
            model.rigid_xyz = model.rigid_rotation = None
            from libgs.metric import psnr, ssim
            rows = []
            for split, dataset in [('train', pipeline.datamodule.get_train_dataset()),
                                   ('test', pipeline.datamodule.get_test_dataset())]:
                for index in range(len(dataset)):
                    view = dataset[index].to(module.device)
                    camera = Path(str(view.path)).name
                    if list(view.image.shape[-2:][::-1]) != protocol['image_size']:
                        raise ValueError('Evaluated image dimensions disagree')
                    key = (split, camera)
                    gt = (tensor_digest(view.image), tensor_digest(view.full_proj_transform))
                    if key in ground_truth and ground_truth[key] != gt:
                        raise ValueError(f'GT or camera changed between checkpoints: {key}')
                    ground_truth[key] = gt
                    with torch.no_grad():
                        model.render_attribute_cache = False
                        off = module.forward(view)['render'].clamp(0., 1.)
                        model.render_attribute_cache = True
                        model._render_cache = {}
                        on = module.forward(view)['render'].clamp(0., 1.)
                        torch.testing.assert_close(on, off, rtol=0, atol=0)
                        rows.append(dict(split=split, camera=camera, gt_sha256=gt[0],
                            projection_sha256=gt[1], cache_max_absolute_difference=0.,
                            psnr=float(psnr(off, view.image)), ssim=float(ssim(off, view.image))))
                    del view, off, on
            means = {split: {metric: statistics.fmean(row[metric] for row in rows if row['split'] == split)
                            for metric in ('psnr', 'ssim')} for split in ('train', 'test')}
            records.append(dict(step=step, gs=len(model.get_xyz),
                active_sh_degree=model.active_sh_degree, max_sh_degree=model.max_sh_degree,
                noise_before=previous_noise, noise_for_evaluation=model.noise_scale,
                means=means, views=rows))
            print(json.dumps(dict(step=step, means=means, cache_exact=True)), flush=True)
            del module, model, dataset, pipeline
            gc.collect()
            torch.cuda.empty_cache()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(protocol=protocol, records=records,
        same_ground_truth_and_camera=True, cache_on_off_images_bitwise_equal=True,
        no_training_updates=True), indent=2))


if __name__ == '__main__':
    main()
