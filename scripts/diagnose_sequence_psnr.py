"""Read-only camera/stage evaluation of an existing sequence's checkpoints.

The source run, parameters and packets are never written. Diagnostic held-out
scores are not used to select training cameras, parameters or checkpoints.
"""
import argparse
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import warnings

os.environ.setdefault('TORCH_CUDA_ARCH_LIST', '8.6')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from omegaconf import OmegaConf
from torchvision.utils import save_image

from libgs.metric import psnr
from libgs.utils.config import to_structured
from pipeline.data import DataModule
from pipeline.entry import Config
from pipeline.module import Module
from pipeline.model.residual_birth import SubsetView


def metrics(image, target):
    difference = image.clamp(0, 1)-target
    return dict(psnr=float(psnr(image.clamp(0, 1), target)),
                global_psnr=float(-10*difference.square().mean().log10()),
                mse=float(difference.square().mean()),
                l1=float(difference.abs().mean()),
                signed_rgb_error=difference.mean((1, 2)).tolist())


def aggregate(rows):
    return {split: {key: sum(row[key] for row in selected)/len(selected)
                    for key in ('psnr', 'global_psnr', 'mse', 'l1')}
            for split in ('train', 'test')
            if (selected := [row for row in rows if row['split'] == split])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frames', nargs='+', type=int, default=[1])
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        config, extra = to_structured(OmegaConf.load(args.run/'config.yaml'), Config)
    if extra:
        raise ValueError(f'Unknown configuration fields: {extra}')
    torch.set_num_threads(4)
    torch.manual_seed(1729)
    data = DataModule(config.data)
    data.setup()
    module = Module(config.module, datamodule=data)
    module.trainer = SimpleNamespace(config=config.trainer, global_step=0,
                                     output_dir=args.output, device=torch.device('cuda'))
    module.to('cuda').setup()
    views = {}
    records = []

    def frame_views(frame):
        if frame not in views:
            data.current_frame = frame
            selected = []
            for split, dataset in (('train', data.get_train_dataset()),
                                   ('test', data.get_test_dataset())):
                for index in sorted(range(len(dataset)), key=lambda i: dataset.items[i].path.name):
                    selected.append((split, dataset[index].to('cuda')))
            views[frame] = selected
        return views[frame]

    def evaluate(frame, step, label, *, old_only=False, final_subset=False):
        rows = []
        for split, view in frame_views(frame):
            if final_subset:
                keep = torch.arange(len(module.gaussians._xyz), device='cuda') < module.frame_report['old_count']
                previous = module.renderer.gaussians
                try:
                    module.renderer.gaussians = SubsetView(module.gaussians, keep)
                    image = module.renderer(view, module.background)['render']
                finally:
                    module.renderer.gaussians = previous
            else:
                image = module.forward(view, old_only=old_only)['render']
            row = dict(camera=view.path.name, split=split, **metrics(image, view.image))
            rows.append(row)
            if split == 'test':
                prefix = args.output/f'frame{frame:03d}_step{step}_{label}'
                save_image(image.clamp(0, 1), str(prefix)+'_render.png')
                save_image(view.image, args.output/f'frame{frame:03d}_gt.png')
                torch.save(dict(render=image.clamp(0, 1).cpu(), target=view.image.cpu()),
                           str(prefix)+'_pixels.pt')
        record = dict(frame=frame, step=step, label=label, cameras=rows,
                      aggregate=aggregate(rows))
        records.append(record)
        (args.output/'stages.json').write_text(json.dumps(dict(
            source_run=str(args.run.resolve()), source_modified=False,
            scope='Diagnostic renders of saved states; no optimizer updates',
            records=records), indent=2))
        print(json.dumps({key: record[key] for key in ('frame', 'step', 'label', 'aggregate')}), flush=True)

    with torch.no_grad():
        checkpoint = torch.load(config.ckpt_path, map_location='cuda', weights_only=False)
        data.current_frame = 0
        module.trainer.global_step = int(checkpoint['global_step'])
        module.on_load_checkpoint(checkpoint)
        evaluate(0, module.trainer.global_step, 'native_first_frame')
        del checkpoint
        for frame in args.frames:
            paths = sorted(args.run.glob(f'rigid-ckpt-{frame}-*.pth'),
                           key=lambda path: int(path.stem.rsplit('-', 1)[1]))
            if not paths:
                raise ValueError(f'No saved full checkpoints for frame {frame}')
            for path in paths:
                checkpoint = torch.load(path, map_location='cuda', weights_only=False)
                data.current_frame = frame
                module.trainer.global_step = step = int(checkpoint['global_step'])
                module.on_load_checkpoint(checkpoint)
                evaluate(frame, step, 'complete')
                if module.motion_committed and not module.frame_finalized:
                    evaluate(frame, step, 'old_only', old_only=True)
                if module.frame_finalized:
                    evaluate(frame, step, 'final_without_newborns', final_subset=True)
                del checkpoint


if __name__ == '__main__':
    main()
