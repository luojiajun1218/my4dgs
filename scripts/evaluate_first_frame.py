"""Evaluate a completed ReCon-GS first frame at training and stream sizes.

The test view is used here for user-requested model selection, so its score is
not an independent estimate of the selected model's generalization quality.
"""

import argparse
import hashlib
import importlib
import json
import math
from pathlib import Path
import sys

import torch
from torchvision.utils import save_image


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--full-root', type=Path, required=True)
    parser.add_argument('--half-root', type=Path, required=True)
    parser.add_argument('--half-only', action='store_true',
                        help='Evaluate only the stream-size first frame.')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--recon-root', type=Path,
                        default=Path('/home/gamer-02/Documents/DeepLearning/ReCon-GS'))
    parser.add_argument('--opacity-threshold', type=float, default=0.0075)
    parser.add_argument('--expected-step', type=int, default=15000,
                        help='Expected frame-0 checkpoint step (default: 15000).')
    parser.add_argument('--expected-full-size', type=int, nargs=2)
    parser.add_argument('--expected-half-size', type=int, nargs=2)
    args = parser.parse_args()
    if not 0 <= args.opacity_threshold < 1:
        parser.error('opacity threshold must be in [0, 1)')
    if args.expected_step <= 0:
        parser.error('expected step must be positive')
    if args.output.exists():
        raise FileExistsError(args.output)
    sys.path.insert(0, str(args.recon_root))
    data_mod = importlib.import_module('pipeline.ReCon-GS.data')
    model_mod = importlib.import_module('pipeline.ReCon-GS.model.gaussian')
    config_mod = importlib.import_module('pipeline.ReCon-GS.module')
    from libgs.metric import psnr, ssim
    from libgs.renderer import Renderer, RendererConfig

    ckpt = torch.load(args.checkpoint, map_location='cuda', weights_only=False)
    step = int(ckpt.get('global_step', -1))
    if int(ckpt.get('current_frame', -1)) != 0 or step != args.expected_step:
        raise ValueError(f'Expected a frame-0 checkpoint at step {args.expected_step}')
    # Preserve the checkpoint's SH capacity when comparing first-frame fits.
    coefficients = int(ckpt['gaussians'][2].shape[1] + ckpt['gaussians'][3].shape[1])
    sh_side = math.isqrt(coefficients)
    if sh_side * sh_side != coefficients:
        raise ValueError(f'Invalid SH coefficient count: {coefficients}')
    sh_degree = sh_side - 1
    active_sh_degree = int(ckpt['gaussians'][0])
    if not 0 <= active_sh_degree <= sh_degree:
        raise ValueError('Active SH degree exceeds checkpoint capacity')
    model = model_mod.GaussianModel(sh_degree)
    model.restore(ckpt['gaussians'], config_mod.GaussianConfig())
    renderer = Renderer(RendererConfig(), model)
    views = {}
    roots = ([('half', args.half_root)] if args.half_only else
             [('full', args.full_root), ('half', args.half_root)])
    for name, root in roots:
        data = data_mod.DataModule(data_mod.DataConfig(
            root=str(root), resolution=-1, split_train_test=True, shuffle=False,
            extra_dataset_kwargs={'num_frames': 1},
            extra_dataloader_kwargs={'num_workers': 1, 'pin_memory': False}))
        data.setup()
        data.current_frame = 0
        test = data.get_test_dataset()
        if len(test) != 1 or test.items[0].path.name != 'cam00.png':
            raise ValueError(f'{name}: expected exactly one held-out cam00 view')
        views[name] = next(iter(data.test_dataloader())).to('cuda')

    def evaluate(view):
        with torch.no_grad():
            predicted = renderer(view, torch.zeros(3, device='cuda'))['render'].clamp(0, 1)
            reference = view.image
            mse = (predicted - reference).square().mean()
            result = dict(
                image_size=[int(predicted.shape[2]), int(predicted.shape[1])],
                project_psnr_db=float(psnr(predicted, reference)),
                global_mse_psnr_db=float(-10 * torch.log10(mse)),
                ssim=float(ssim(predicted, reference)),
                l1=float((predicted - reference).abs().mean()),
            )
        return result, predicted

    expected = {'full': args.expected_full_size, 'half': args.expected_half_size}
    initial_count = int(model.get_xyz.shape[0])
    raw = {}
    for name, view in views.items():
        raw[name], predicted = evaluate(view)
        if expected[name] is not None and raw[name]['image_size'] != expected[name]:
            raise ValueError(f'{name}: unexpected image size {raw[name]["image_size"]}')
        if name == 'half':
            preview = args.output.with_name('frame0_cam00_half.png')
            if preview.exists():
                raise FileExistsError(preview)
            save_image(predicted, preview)
    with torch.no_grad():
        removed = (model.get_opacity.flatten() < args.opacity_threshold)
        removed_count = int(removed.sum())
        if removed_count == initial_count:
            raise ValueError('Opacity threshold would remove every point')
        if removed_count:
            model.prune_points(removed)
    pruned = {}
    for name, view in views.items():
        pruned[name], _ = evaluate(view)
    report = {
        'checkpoint': str(args.checkpoint.resolve()),
        'checkpoint_sha256': file_sha256(args.checkpoint),
        'frame': 0, 'steps': step, 'heldout_camera': 'cam00.png',
        'active_sh_degree': active_sh_degree, 'max_sh_degree': sh_degree,
        'evaluation_scope': 'half_only' if args.half_only else 'full_and_half',
        'selection_metric': 'raw half-resolution project PSNR',
        'psnr_definition': 'project PSNR averages per-channel dB; global MSE PSNR is also recorded',
        'selection_caveat': 'cam00 participates in model selection and is not an independent held-out result',
        'raw_point_count': initial_count,
        'raw': raw,
        'opacity_threshold': args.opacity_threshold,
        'points_removed_at_threshold': removed_count,
        'pruned_point_count': int(model.get_xyz.shape[0]),
        'after_initial_opacity_prune': pruned,
        'half_render': str(args.output.with_name('frame0_cam00_half.png').resolve()),
    }
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
