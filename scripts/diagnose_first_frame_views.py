"""Compare existing native first frames across cameras and resolutions, read only."""
import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('TORCH_CUDA_ARCH_LIST', '8.6')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torchvision.utils import save_image
from libgs.renderer import Renderer, RendererConfig
from pipeline.data import DataConfig, DataModule
from pipeline.model.gaussian import GaussianModel
from pipeline.module import GaussianConfig
from scripts.diagnose_sequence_psnr import metrics, aggregate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', nargs='+', type=Path, required=True)
    parser.add_argument('--half-root', type=Path, required=True)
    parser.add_argument('--full-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    torch.set_num_threads(4)
    datasets = {}
    for label, root in (('half', args.half_root), ('full', args.full_root)):
        # For these >1600px full inputs, explicit 1600 is numerically the
        # same resize as -1, without libgs's worker-only warning callback.
        data = DataModule(DataConfig(root=str(root), resolution=1 if label == 'half' else 1600,
                                    shuffle=False, extra_dataset_kwargs={'num_frames': 1}))
        data.setup()
        data.current_frame = 0
        datasets[label] = [(split, dataset, index) for split, dataset in
                           (('train', data.get_train_dataset()), ('test', data.get_test_dataset()))
                           for index in range(len(dataset))]
    results = []
    for index, path in enumerate(args.checkpoint):
        checkpoint = torch.load(path, map_location='cuda', weights_only=False)
        if int(checkpoint.get('current_frame', -1)) != 0 or 'rigid_tree_version' in checkpoint:
            raise ValueError('Expected native first-frame checkpoints')
        model = GaussianModel(1)
        model.restore(checkpoint['gaussians'], GaussianConfig())
        model.prune_points(model.get_opacity.flatten() < .0075)
        renderer = Renderer(RendererConfig(), model)
        for size, items in datasets.items():
            rows = []
            with torch.no_grad():
                for split, dataset, row_index in items:
                    view = dataset[row_index].to('cuda')
                    image = renderer(view, torch.zeros(3, device='cuda'))['render'].clamp(0, 1)
                    rows.append(dict(camera=view.path.name, split=split,
                                     image_size=list(reversed(image.shape[-2:])),
                                     **metrics(image, view.image)))
                    if view.path.name in ('cam00.png', 'cam13.png', 'cam10.png'):
                        stem = args.output/f'model{index}_{size}_{view.path.stem}'
                        save_image(image, str(stem)+'_render.png')
                        save_image(view.image, str(stem)+'_gt.png')
            record = dict(checkpoint=str(path.resolve()), model_index=index, size=size,
                          point_count=len(model._xyz), step=checkpoint['global_step'],
                          cameras=rows, aggregate=aggregate(rows))
            results.append(record)
            (args.output/'results.json').write_text(json.dumps(dict(
                source_modified=False, optimizer_updates=0, results=results), indent=2))
            print(json.dumps({key: record[key] for key in
                              ('model_index', 'size', 'point_count', 'aggregate')}), flush=True)
        del checkpoint, model, renderer


if __name__ == '__main__':
    main()
