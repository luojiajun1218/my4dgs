"""Evaluate existing frame-zero checkpoints without further training."""
import argparse
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from libgs.data.dataset.colmap import Colmap4DDataset
from libgs.metric import psnr, ssim
from libgs.renderer import Renderer, RendererConfig


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resolution', type=float, help='Optional evaluation-only resolution control')
    parser.add_argument('--prune-oversized', action='store_true', help='Optional in-memory size-pruning diagnostic')
    parser.add_argument('checkpoints', nargs='+', type=Path)
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    if args.resolution is not None:
        config.data.resolution = args.resolution
    dataset_kwargs = dict(config.data.extra_dataset_kwargs)
    dataset_kwargs['num_frames'] = 1
    dataset = Colmap4DDataset(Path(config.data.root), 'none', config.data.resolution, **dataset_kwargs)
    dataset.setup()
    test_dataset = Colmap4DDataset(Path(config.data.root), 'test', config.data.resolution, **dataset_kwargs)
    test_dataset.setup()
    test_names = {item.path.name for item in test_dataset.items}
    viewpoints = [dataset[i].to('cuda') for i in range(len(dataset))]
    module = importlib.import_module('pipeline.ReCon-GS.module')
    model = importlib.import_module('pipeline.ReCon-GS.model.gaussian')
    background = torch.full((3,), float(config.data.white_background), device='cuda')
    results = []
    for path in args.checkpoints:
        checkpoint = torch.load(path, map_location='cuda', weights_only=False)
        if checkpoint['current_frame'] != 0:
            raise ValueError(f'{path} is not a first-frame checkpoint')
        gaussian = model.GaussianModel(config.module.sh_degree)
        gaussian.restore(checkpoint['gaussians'], module.GaussianConfig(**dict(config.module.gaussian)))
        original_count = int(gaussian.get_xyz.shape[0])
        if args.prune_oversized:
            gaussian.prune_points(gaussian.get_scaling.max(dim=1).values > .1 * gaussian.spatial_lr_scale)
        renderer = Renderer(RendererConfig(**dict(config.module.renderer)), gaussian)
        views = []
        for item, viewpoint in zip(dataset.items, viewpoints):
            rendered = renderer(viewpoint, background)['render'].clamp(0, 1)
            views.append(dict(name=item.path.name, source=str(item.path.resolve()), is_test=item.path.name in test_names,
                              size=list(viewpoint.image.shape[1:][::-1]),
                              psnr=float(psnr(rendered, viewpoint.image).mean()),
                              ssim=float(ssim(rendered, viewpoint.image))))
        opacity = gaussian.get_opacity.squeeze()
        max_scale = gaussian.get_scaling.max(dim=1).values
        row = dict(checkpoint=str(path), step=checkpoint['global_step'],
                   original_gs_count=original_count, evaluation_only_pruning=args.prune_oversized,
                   gs_count=int(gaussian.get_xyz.shape[0]), sh_degree=gaussian.active_sh_degree,
                   scene_extent=gaussian.spatial_lr_scale,
                   scale_quantiles=max_scale.quantile(torch.tensor([.5, .9, .99], device='cuda')).tolist(),
                   opacity_quantiles=opacity.quantile(torch.tensor([.5, .9, .99], device='cuda')).tolist(),
                   opacity_above_099=float((opacity > .99).float().mean()),
                   oversized_count=int((max_scale > .1 * gaussian.spatial_lr_scale).sum()),
                   test_psnr=float(np.mean([v['psnr'] for v in views if v['is_test']])),
                   test_ssim=float(np.mean([v['ssim'] for v in views if v['is_test']])),
                   train_mean_psnr=float(np.mean([v['psnr'] for v in views if not v['is_test']])), views=views)
        results.append(row)
        print(json.dumps({k: v for k, v in row.items() if k != 'views'}), flush=True)
        del renderer, gaussian, checkpoint
        torch.cuda.empty_cache()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
