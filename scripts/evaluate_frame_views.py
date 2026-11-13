"""检查指定帧在各训练和测试相机下的画质。

从实验码流或指定帧检查点读取 GS，按训练、测试相机分别渲染并汇总画质；
支持对比数据配置、透明度裁剪，并可输出渲染图和报告。裁剪只作用于评估副本。
本文件不训练模型，也不负责修改源检查点或码流。
"""

import argparse
import json
import math
from pathlib import Path
import sys

import torch
from omegaconf import OmegaConf
from torchvision.utils import save_image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from libgs.metric import psnr, ssim
from libgs.renderer import Renderer
from libgs.utils.config import to_structured
from pipeline.data import DataModule
from pipeline.entry import Config
from pipeline.model.frame_stream_decode import replay
from pipeline.model.frame_stream_format import RAW_ATTRIBUTES, RAW_FIELDS
from pipeline.model.gaussian import GaussianModel

CAPTURE_INDEX = dict(xyz=1, sh_dc=2, sh_rest=3, logscale=4, rotation=5, opacity=6)


def _load_config(path, data_root):
    config, extra = to_structured(OmegaConf.load(path), Config)
    if extra:
        raise ValueError(f'Unknown configuration fields: {extra}')
    if data_root is not None:
        config.data.root = str(data_root)
    config.data.shuffle = False
    return config


def _load_checkpoint(path, frame):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if checkpoint.get('current_frame') != frame:
        raise ValueError('Checkpoint frame does not match --frame')
    if frame > 0 and not (checkpoint.get('rigid_tree_frame') or {}).get('finalized'):
        raise ValueError('Dynamic checkpoint must be a finalized frame')
    capture = checkpoint['gaussians']
    raw = {name: capture[index].detach().cpu() for name, index in CAPTURE_INDEX.items()}
    return raw, int(capture[0]), checkpoint.get('global_step')


def _make_model(raw, active_degree, device):
    rest = raw['sh_rest'].shape[1]
    side = math.isqrt(rest + 1)
    if side * side != rest + 1 or not 0 <= active_degree < side:
        raise ValueError('Invalid SH coefficient count or active degree')
    model = GaussianModel(side - 1)
    model.active_sh_degree = active_degree
    for name, attribute in zip(RAW_FIELDS, RAW_ATTRIBUTES):
        setattr(model, attribute, torch.nn.Parameter(raw[name].to(device), requires_grad=False))
    return model


def _metrics(image, target):
    difference = image - target
    mse = difference.square().mean()
    return dict(psnr=float(psnr(image, target)), global_psnr=float(-10 * mse.log10()),
                ssim=float(ssim(image, target)),
                mse=float(mse), l1=float(difference.abs().mean()),
                signed_rgb_error=difference.mean((1, 2)).tolist())


def _aggregate(rows):
    result = {}
    for split in ('train', 'test'):
        selected = [row for row in rows if row['split'] == split]
        if selected:
            result[split] = {name: sum(row[name] for row in selected) / len(selected)
                             for name in ('psnr', 'global_psnr', 'ssim', 'mse', 'l1')}
            result[split]['camera_count'] = len(selected)
    return result


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--run', type=Path, help='Experiment directory containing the frame stream')
    source.add_argument('--checkpoint', type=Path, help='Checkpoint for the selected frame')
    parser.add_argument('--config', type=Path, help='Required with --checkpoint; defaults to run/config.yaml')
    parser.add_argument('--compare-config', type=Path, action='append', default=[],
                        help='Additional data/render configuration for comparing resolutions; may be repeated')
    parser.add_argument('--frame', type=int, default=0, help='Frame number, starting at 0')
    parser.add_argument('--data-root', type=Path, help='Override the data root recorded in the config')
    parser.add_argument('--opacity-threshold', type=float,
                        help='Also evaluate a copy with GS below this opacity removed (0 <= threshold < 1)')
    parser.add_argument('--output', type=Path, required=True, help='New directory for the diagnostic report')
    parser.add_argument('--save-images', action='store_true', help='Save rendered and target images for every view')
    args = parser.parse_args()
    if args.frame < 0:
        parser.error('--frame must be nonnegative')
    if args.checkpoint is not None and args.config is None:
        parser.error('--checkpoint requires --config')
    if args.opacity_threshold is not None and not 0 <= args.opacity_threshold < 1:
        parser.error('--opacity-threshold must be in [0, 1)')
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    if args.run is not None and args.output.resolve().is_relative_to(args.run.resolve()):
        raise ValueError('Write the diagnostic outside the source experiment directory')
    if args.checkpoint is not None and args.output.resolve().is_relative_to(args.checkpoint.resolve().parent):
        raise ValueError('Write the diagnostic outside the checkpoint directory')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for rendering')

    config_path = args.config or args.run / 'config.yaml'
    config = _load_config(config_path, args.data_root)
    device = torch.device('cuda')

    if args.run is not None:
        stream = args.run / 'frame_storage' / 'rigid_tree'
        state, raw = replay(stream, stream / 'reference_frame0.npz', args.frame)
        if state.frame_index != args.frame:
            raise ValueError('Decoded frame does not match --frame')
        active_degree = config.module.sh_degree
        source_info = dict(kind='decoded_stream', run=str(args.run.resolve()))
    else:
        raw, active_degree, step = _load_checkpoint(args.checkpoint, args.frame)
        source_info = dict(kind='checkpoint', checkpoint=str(args.checkpoint.resolve()), step=step)

    variants = [('original', raw)]
    if args.opacity_threshold is not None:
        keep = raw['opacity'].sigmoid().flatten() >= args.opacity_threshold
        if not bool(keep.any()):
            raise ValueError('Opacity threshold would remove every Gaussian')
        variants.append(('opacity_pruned', {name: value[keep] for name, value in raw.items()}))

    args.output.mkdir(parents=True)
    datasets = []
    configs = [(config_path, config)] + [(path, _load_config(path, None)) for path in args.compare_config]
    for config_index, (path, current_config) in enumerate(configs):
        data = DataModule(current_config.data)
        data.setup()
        if args.frame >= data.num_frames:
            raise ValueError(f'Frame {args.frame} is outside the dataset of {data.num_frames} frames: {path}')
        data.current_frame = args.frame
        background = torch.full((3,), 1. if current_config.data.white_background else 0., device=device)
        variant_results = []
        for label, attributes in variants:
            model = _make_model(attributes, active_degree, device)
            renderer = Renderer(current_config.module.renderer, model)
            rows = []
            for split, dataset in (('train', data.get_train_dataset()), ('test', data.get_test_dataset())):
                if dataset is None:
                    continue
                for index in sorted(range(len(dataset)), key=lambda i: str(dataset.items[i].path.name)):
                    view = dataset[index].to(device)
                    expected = current_config.data.expected_image_size
                    if expected is not None and tuple(view.image.shape[-2:]) != (expected[1], expected[0]):
                        raise ValueError(f'{view.path.name} image size does not match {path}')
                    image = renderer(view, background)['render'].clamp(0, 1)
                    if image.shape != view.image.shape or not bool(torch.isfinite(image).all()):
                        raise ValueError(f'Invalid render for {view.path.name}')
                    row = dict(camera=view.path.name, split=split,
                               image_size=[int(image.shape[2]), int(image.shape[1])],
                               **_metrics(image, view.image))
                    rows.append(row)
                    if args.save_images:
                        stem = args.output / f'data{config_index:02d}_{label}_{split}_{view.path.stem}'
                        save_image(image, str(stem) + '_render.png')
                        save_image(view.image, str(stem) + '_gt.png')
                    print(f"data{config_index:02d} {label} {split} {view.path.name}: "
                          f"PSNR {row['psnr']:.4f} dB", flush=True)
            if not rows:
                raise ValueError(f'No camera views found for the selected frame: {path}')
            variant_results.append(dict(variant=label, point_count=len(attributes['xyz']),
                                        cameras=rows, aggregate=_aggregate(rows)))
            del renderer, model
        datasets.append(dict(config=str(path.resolve()),
                             data_root=str(Path(current_config.data.root).resolve()),
                             variants=variant_results))

    report = dict(frame=args.frame, source=source_info, active_sh_degree=active_degree,
                  opacity_threshold=args.opacity_threshold, datasets=datasets,
                  comparison_scope='Pruning is applied to the selected frame state only; '
                                   'data configurations are evaluated independently')
    (args.output / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f"Saved {args.output / 'results.json'}", flush=True)


if __name__ == '__main__':
    main()
